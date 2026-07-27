import CoreGraphics
import Accelerate
import UIKit

/// Minimal RGB(A)/gray raw-pixel access shared by BackgroundReconstructor,
/// LightmapExtractor, and Compositor -- avoids re-deriving CGImage
/// byte-layout math (bytesPerRow/bytesPerPixel offsets) in three places.
/// Uses vImage (Accelerate) buffers so the per-pixel math in the alignment
/// pyramid and trimmed-mean stays vectorized rather than falling back to
/// pure Swift loops over millions of pixels -- same performance concern
/// the Python reference's trimmed_mean_vectorized() docstring already
/// flagged (a plain Python/Swift nested loop over N frames x H x W pixels
/// is not feasible; needs SIMD/vDSP-style batch ops).
struct RGBBuffer {
    var r: [Float]
    var g: [Float]
    var b: [Float]
    let width: Int
    let height: Int

    static func from(cgImage: CGImage) -> RGBBuffer {
        let w = cgImage.width
        let h = cgImage.height
        var rArr = [Float](repeating: 0, count: w * h)
        var gArr = [Float](repeating: 0, count: w * h)
        var bArr = [Float](repeating: 0, count: w * h)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var raw = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &raw, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return RGBBuffer(r: rArr, g: gArr, b: bArr, width: w, height: h)
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        for i in 0..<(w * h) {
            rArr[i] = Float(raw[i * 4])
            gArr[i] = Float(raw[i * 4 + 1])
            bArr[i] = Float(raw[i * 4 + 2])
        }
        return RGBBuffer(r: rArr, g: gArr, b: bArr, width: w, height: h)
    }

    /// Standard luma (matches cv2.COLOR_BGR2GRAY's coefficients), used only
    /// for the alignment pyramid's SAD search -- same role as the Python
    /// reference's cv2.cvtColor(..., BGR2GRAY) step.
    func grayscale() -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            out[i] = 0.114 * b[i] + 0.587 * g[i] + 0.299 * r[i]
        }
        return out
    }

    func toCGImage() -> CGImage? {
        var raw = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            raw[i * 4] = UInt8(max(0, min(255, r[i].rounded())))
            raw[i * 4 + 1] = UInt8(max(0, min(255, g[i].rounded())))
            raw[i * 4 + 2] = UInt8(max(0, min(255, b[i].rounded())))
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(raw) as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}

/// Binary person/hole mask, sourced from the Tier1 segmentation output
/// (mask.mp4 convention: white=person/hole, matching Android/LamaRunner's
/// existing convention). true = person (excluded from background
/// aggregation), matching the Python reference's bool_masks.
struct MaskBuffer {
    var isPerson: [Bool]
    let width: Int
    let height: Int

    static func from(cgImage: CGImage, threshold: UInt8 = 127) -> MaskBuffer {
        let w = cgImage.width
        let h = cgImage.height
        var raw = [UInt8](repeating: 0, count: w * h)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(
            data: &raw, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return MaskBuffer(isPerson: [Bool](repeating: false, count: w * h), width: w, height: h)
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        let mask = raw.map { $0 > threshold }
        return MaskBuffer(isPerson: mask, width: w, height: h)
    }
}
