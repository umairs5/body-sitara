import CoreML
import CoreImage
import UIKit

/// RIFE (IFNet) inference wrapper -- mirrors RifeInterpolator.kt's
/// structure (per-stage timing split: tensor build / model run /
/// postprocess), so iPhone numbers are directly comparable to the
/// existing Android measurements at the same granularity.
///
/// Model: RifeIFNet.mlpackage, converted from this project's own
/// rife_ifnet.onnx export (ios_bench/convert_models.py) -- same model,
/// same fixed 1280x1280 padded input, as the Android RifeInterpolator.
final class RifeRunner {
    private let model: MLModel
    static let requiredSize = 1280 // matches RifeInterpolator.kt's REQUIRED_SIZE

    struct StageTiming {
        let buildMs: Double
        let runMs: Double
        let postprocessMs: Double
    }

    init(configuration: MLModelConfiguration) throws {
        guard let url = Bundle.main.url(forResource: "RifeIFNet", withExtension: "mlmodelc") else {
            fatalError("RifeIFNet.mlmodelc not found in app bundle -- check convert_models.py ran and Xcode build phase copied it in")
        }
        self.model = try MLModel(contentsOf: url, configuration: configuration)
    }

    /// Runs one interpolation, returning the output image and per-stage
    /// timing. frameA/frameB must be the same size; internally padded to
    /// the model's fixed 1280x1280 input and cropped back after inference
    /// -- same convention as RifeInterpolator.kt.
    func interpolateMidpoint(frameA: CGImage, frameB: CGImage) throws -> (UIImage, StageTiming) {
        let w = frameA.width
        let h = frameA.height
        precondition(frameB.width == w && frameB.height == h, "frameA/frameB must be the same size")

        let tBuildStart = CFAbsoluteTimeGetCurrent()
        let tensorA = try Self.cgImageToPaddedCHWTensor(frameA, targetSize: Self.requiredSize)
        let tensorB = try Self.cgImageToPaddedCHWTensor(frameB, targetSize: Self.requiredSize)
        let buildMs = (CFAbsoluteTimeGetCurrent() - tBuildStart) * 1000

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "img0": MLFeatureValue(multiArray: tensorA),
            "img1": MLFeatureValue(multiArray: tensorB),
        ])

        let tRunStart = CFAbsoluteTimeGetCurrent()
        let output = try model.prediction(from: input)
        let runMs = (CFAbsoluteTimeGetCurrent() - tRunStart) * 1000

        let tPostStart = CFAbsoluteTimeGetCurrent()
        guard let outArray = output.featureValue(for: "interpolated")?.multiArrayValue else {
            throw NSError(domain: "RifeRunner", code: 1, userInfo: [NSLocalizedDescriptionKey: "missing 'interpolated' output"])
        }
        let outImage = try Self.chwTensorToImageCropped(outArray, width: w, height: h)
        let postprocessMs = (CFAbsoluteTimeGetCurrent() - tPostStart) * 1000

        return (outImage, StageTiming(buildMs: buildMs, runMs: runMs, postprocessMs: postprocessMs))
    }

    // MARK: - Tensor conversion (pad to fixed size, matching Android's bitmapToPaddedCHWTensor)

    private static func cgImageToPaddedCHWTensor(_ image: CGImage, targetSize: Int) throws -> MLMultiArray {
        let w = image.width
        let h = image.height
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: targetSize), NSNumber(value: targetSize)], dataType: .float32)

        guard let dataProvider = image.dataProvider, let pixelData = dataProvider.data,
              let ptr = CFDataGetBytePtr(pixelData) else {
            throw NSError(domain: "RifeRunner", code: 2, userInfo: [NSLocalizedDescriptionKey: "could not read CGImage pixel data"])
        }
        let bytesPerPixel = image.bitsPerPixel / 8
        let bytesPerRow = image.bytesPerRow

        let bufferPtr = array.dataPointer.bindMemory(to: Float32.self, capacity: 3 * targetSize * targetSize)
        // Zero-fill (padding region), then write real pixels into [0,h)x[0,w).
        bufferPtr.update(repeating: 0, count: 3 * targetSize * targetSize)

        for y in 0..<h {
            for x in 0..<w {
                let pixelOffset = y * bytesPerRow + x * bytesPerPixel
                let r = Float32(ptr[pixelOffset]) / 255.0
                let g = Float32(ptr[pixelOffset + 1]) / 255.0
                let b = Float32(ptr[pixelOffset + 2]) / 255.0
                bufferPtr[0 * targetSize * targetSize + y * targetSize + x] = r
                bufferPtr[1 * targetSize * targetSize + y * targetSize + x] = g
                bufferPtr[2 * targetSize * targetSize + y * targetSize + x] = b
            }
        }
        return array
    }

    private static func chwTensorToImageCropped(_ tensor: MLMultiArray, width: Int, height: Int) throws -> UIImage {
        let targetSize = requiredSize
        let ptr = tensor.dataPointer.bindMemory(to: Float32.self, capacity: 3 * targetSize * targetSize)

        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let r = ptr[0 * targetSize * targetSize + y * targetSize + x]
                let g = ptr[1 * targetSize * targetSize + y * targetSize + x]
                let b = ptr[2 * targetSize * targetSize + y * targetSize + x]
                let idx = (y * width + x) * 4
                pixelData[idx] = UInt8(max(0, min(255, r * 255)))
                pixelData[idx + 1] = UInt8(max(0, min(255, g * 255)))
                pixelData[idx + 2] = UInt8(max(0, min(255, b * 255)))
                pixelData[idx + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent) else {
            throw NSError(domain: "RifeRunner", code: 3, userInfo: [NSLocalizedDescriptionKey: "failed to build output CGImage"])
        }
        return UIImage(cgImage: cgImage)
    }
}
