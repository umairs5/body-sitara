import CoreML
import UIKit

/// big-LaMa (background inpainting) inference wrapper -- same per-stage
/// timing discipline as RifeRunner.swift.
///
/// Model: BigLama.mlpackage, converted from a fresh 1280x1280 ONNX
/// re-export of models/big-lama.pt (ios_bench/convert_models.py) -- same
/// checkpoint the Android LamaDilatedFiller.kt uses, but re-exported at
/// 1280x1280 (matching RIFE's resolution) rather than reusing the
/// existing 512x512 lama_dilated.onnx export, so both models in this
/// benchmark run at the same, real pipeline-relevant resolution.
final class LamaRunner {
    private let model: MLModel
    static let inputSize = 1280 // matches convert_models.py's BENCH_SIZE

    struct StageTiming {
        let buildMs: Double
        let runMs: Double
        let postprocessMs: Double
    }

    init(configuration: MLModelConfiguration) throws {
        guard let url = Bundle.main.url(forResource: "BigLama", withExtension: "mlmodelc") else {
            fatalError("BigLama.mlmodelc not found in app bundle -- check convert_models.py ran and Xcode build phase copied it in")
        }
        self.model = try MLModel(contentsOf: url, configuration: configuration)
    }

    /// Fills the masked region of `image` using `mask` (same white=hole
    /// convention as the Android LamaDilatedFiller). image/mask must be
    /// inputSize x inputSize -- caller resizes/pads before calling, same
    /// as the real pipeline's _resize_square step.
    ///
    /// Also exposed via `fillPixels(background:neverRevealed:)` below for
    /// the real-clip Background Reconstruction pipeline (Tier2-mobile
    /// system-cost benchmark), which works in RGBBuffer/[Bool] space
    /// rather than CGImage, matching BackgroundReconstructor's output.
    func fill(image: CGImage, mask: CGImage) throws -> (UIImage, StageTiming) {
        precondition(image.width == Self.inputSize && image.height == Self.inputSize, "image must be pre-resized to \(Self.inputSize)x\(Self.inputSize)")
        precondition(mask.width == Self.inputSize && mask.height == Self.inputSize, "mask must be pre-resized to \(Self.inputSize)x\(Self.inputSize)")

        let tBuildStart = CFAbsoluteTimeGetCurrent()
        let imageTensor = try Self.cgImageToCHWTensor(image)
        let maskTensor = try Self.cgImageToMaskTensor(mask)
        let buildMs = (CFAbsoluteTimeGetCurrent() - tBuildStart) * 1000

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: imageTensor),
            "mask": MLFeatureValue(multiArray: maskTensor),
        ])

        let tRunStart = CFAbsoluteTimeGetCurrent()
        let output = try model.prediction(from: input)
        let runMs = (CFAbsoluteTimeGetCurrent() - tRunStart) * 1000

        let tPostStart = CFAbsoluteTimeGetCurrent()
        guard let outArray = output.featureValue(for: "output")?.multiArrayValue else {
            throw NSError(domain: "LamaRunner", code: 1, userInfo: [NSLocalizedDescriptionKey: "missing 'output' output"])
        }
        let outImage = try Self.chwTensorToImage(outArray, size: Self.inputSize)
        let postprocessMs = (CFAbsoluteTimeGetCurrent() - tPostStart) * 1000

        return (outImage, StageTiming(buildMs: buildMs, runMs: runMs, postprocessMs: postprocessMs))
    }

    // MARK: - Tensor conversion

    private static func cgImageToCHWTensor(_ image: CGImage) throws -> MLMultiArray {
        let size = inputSize
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        guard let dataProvider = image.dataProvider, let pixelData = dataProvider.data,
              let ptr = CFDataGetBytePtr(pixelData) else {
            throw NSError(domain: "LamaRunner", code: 2, userInfo: [NSLocalizedDescriptionKey: "could not read CGImage pixel data"])
        }
        let bytesPerPixel = image.bitsPerPixel / 8
        let bytesPerRow = image.bytesPerRow
        let bufferPtr = array.dataPointer.bindMemory(to: Float32.self, capacity: 3 * size * size)

        for y in 0..<size {
            for x in 0..<size {
                let offset = y * bytesPerRow + x * bytesPerPixel
                bufferPtr[0 * size * size + y * size + x] = Float32(ptr[offset]) / 255.0
                bufferPtr[1 * size * size + y * size + x] = Float32(ptr[offset + 1]) / 255.0
                bufferPtr[2 * size * size + y * size + x] = Float32(ptr[offset + 2]) / 255.0
            }
        }
        return array
    }

    private static func cgImageToMaskTensor(_ mask: CGImage) throws -> MLMultiArray {
        let size = inputSize
        let array = try MLMultiArray(shape: [1, 1, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        guard let dataProvider = mask.dataProvider, let pixelData = dataProvider.data,
              let ptr = CFDataGetBytePtr(pixelData) else {
            throw NSError(domain: "LamaRunner", code: 3, userInfo: [NSLocalizedDescriptionKey: "could not read mask CGImage pixel data"])
        }
        let bytesPerPixel = mask.bitsPerPixel / 8
        let bytesPerRow = mask.bytesPerRow
        let bufferPtr = array.dataPointer.bindMemory(to: Float32.self, capacity: size * size)

        for y in 0..<size {
            for x in 0..<size {
                let offset = y * bytesPerRow + x * bytesPerPixel
                bufferPtr[y * size + x] = Float32(ptr[offset]) / 255.0 // white=hole, matching Android convention
            }
        }
        return array
    }

    private static func chwTensorToImage(_ tensor: MLMultiArray, size: Int) throws -> UIImage {
        let ptr = tensor.dataPointer.bindMemory(to: Float32.self, capacity: 3 * size * size)
        var pixelData = [UInt8](repeating: 0, count: size * size * 4)
        for y in 0..<size {
            for x in 0..<size {
                let r = ptr[0 * size * size + y * size + x]
                let g = ptr[1 * size * size + y * size + x]
                let b = ptr[2 * size * size + y * size + x]
                let idx = (y * size + x) * 4
                pixelData[idx] = UInt8(max(0, min(255, r * 255)))
                pixelData[idx + 1] = UInt8(max(0, min(255, g * 255)))
                pixelData[idx + 2] = UInt8(max(0, min(255, b * 255)))
                pixelData[idx + 3] = 255
            }
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(
                width: size, height: size,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: size * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent) else {
            throw NSError(domain: "LamaRunner", code: 4, userInfo: [NSLocalizedDescriptionKey: "failed to build output CGImage"])
        }
        return UIImage(cgImage: cgImage)
    }

    /// Core-fill entry point for the real Background Reconstruction
    /// pipeline: fills ONLY the never-revealed pixels of `plate`, once per
    /// clip -- matching the Python reference's run_lama_fill() (resize to
    /// model's fixed square input, infer, paste back only the
    /// never-revealed region at original resolution) and Danial's
    /// confirmed once-per-clip design (not per-frame).
    func fillPixels(plate: RGBBuffer, neverRevealed: [Bool]) throws -> (RGBBuffer, StageTiming) {
        guard let plateImage = plate.toCGImage() else {
            throw NSError(domain: "LamaRunner", code: 5, userInfo: [NSLocalizedDescriptionKey: "failed to rasterize plate"])
        }
        let maskImage = Self.maskToCGImage(neverRevealed, width: plate.width, height: plate.height)

        let resizedImage = Self.resizeSquare(plateImage, to: Self.inputSize)
        let resizedMask = Self.resizeSquare(maskImage, to: Self.inputSize)

        let (filledUIImage, timing) = try fill(image: resizedImage, mask: resizedMask)
        guard let filledCG = filledUIImage.cgImage else {
            throw NSError(domain: "LamaRunner", code: 6, userInfo: [NSLocalizedDescriptionKey: "no cgImage on LaMa output"])
        }
        let filledFull = Self.resizeSquareBackToOriginal(filledCG, width: plate.width, height: plate.height)
        let filledBuf = RGBBuffer.from(cgImage: filledFull)

        var outR = plate.r, outG = plate.g, outB = plate.b
        for i in 0..<(plate.width * plate.height) where neverRevealed[i] {
            outR[i] = filledBuf.r[i]
            outG[i] = filledBuf.g[i]
            outB[i] = filledBuf.b[i]
        }
        return (RGBBuffer(r: outR, g: outG, b: outB, width: plate.width, height: plate.height), timing)
    }

    private static func maskToCGImage(_ neverRevealed: [Bool], width: Int, height: Int) -> CGImage {
        var raw = [UInt8](repeating: 0, count: width * height)
        for i in 0..<(width * height) { raw[i] = neverRevealed[i] ? 255 : 0 }
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let provider = CGDataProvider(data: Data(raw) as CFData)!
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
            bytesPerRow: width, space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )!
    }

    /// Resizes to a size x size square. Always rasterizes through an RGBA
    /// context (rather than passing the source image's own bitmapInfo
    /// through, which is fragile for grayscale-no-alpha sources) -- safe
    /// for both the RGB plate and the single-channel mask image, since
    /// CGContext.draw() handles the colorspace conversion either way.
    private static func resizeSquare(_ image: CGImage, to size: Int) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: size, height: size,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return ctx.makeImage()!
    }

    private static func resizeSquareBackToOriginal(_ image: CGImage, width: Int, height: Int) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()!
    }
}
