import Accelerate
import CoreGraphics

/// Swift port of Danial's lightmap extraction (BackgroundInpaint.kt phase
/// 1b, per Tier_2_Mobile_Comprehensive_Guide.md): "Creates a low-frequency
/// illumination map by downscaling to 48x48, applying a pure-Kotlin
/// separable Gaussian blur (r=2, sigma=3), and upscaling." Operates on the
/// reconstructed background (output of BackgroundReconstructor + LaMa),
/// not on a raw frame.
enum LightmapExtractor {
    static let downscaleSize = 48
    static let blurRadius = 2
    static let sigma: Float = 3.0

    struct Result {
        let lightmap: RGBBuffer
        let totalMs: Double
    }

    static func extract(from background: RGBBuffer) -> Result {
        let t0 = CFAbsoluteTimeGetCurrent()

        let smallR = downscale(background.r, width: background.width, height: background.height, newSize: downscaleSize)
        let smallG = downscale(background.g, width: background.width, height: background.height, newSize: downscaleSize)
        let smallB = downscale(background.b, width: background.width, height: background.height, newSize: downscaleSize)

        let kernel = gaussianKernel1D(radius: blurRadius, sigma: sigma)
        let blurredR = separableBlur(smallR, size: downscaleSize, kernel: kernel)
        let blurredG = separableBlur(smallG, size: downscaleSize, kernel: kernel)
        let blurredB = separableBlur(smallB, size: downscaleSize, kernel: kernel)

        let upR = upscale(blurredR, fromSize: downscaleSize, toW: background.width, toH: background.height)
        let upG = upscale(blurredG, fromSize: downscaleSize, toW: background.width, toH: background.height)
        let upB = upscale(blurredB, fromSize: downscaleSize, toW: background.width, toH: background.height)

        let totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let lightmap = RGBBuffer(r: upR, g: upG, b: upB, width: background.width, height: background.height)
        return Result(lightmap: lightmap, totalMs: totalMs)
    }

    private static func gaussianKernel1D(radius: Int, sigma: Float) -> [Float] {
        let size = 2 * radius + 1
        var kernel = [Float](repeating: 0, count: size)
        var sum: Float = 0
        for i in 0..<size {
            let x = Float(i - radius)
            let v = exp(-(x * x) / (2 * sigma * sigma))
            kernel[i] = v
            sum += v
        }
        for i in 0..<size { kernel[i] /= sum }
        return kernel
    }

    private static func downscale(_ src: [Float], width: Int, height: Int, newSize: Int) -> [Float] {
        var srcBuf = src
        var dstBuf = [Float](repeating: 0, count: newSize * newSize)
        var srcVImage = vImage_Buffer(data: &srcBuf, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width * MemoryLayout<Float>.size)
        var dstVImage = vImage_Buffer(data: &dstBuf, height: vImagePixelCount(newSize), width: vImagePixelCount(newSize), rowBytes: newSize * MemoryLayout<Float>.size)
        vImageScale_PlanarF(&srcVImage, &dstVImage, nil, vImage_Flags(kvImageHighQualityResampling))
        return dstBuf
    }

    private static func upscale(_ src: [Float], fromSize: Int, toW: Int, toH: Int) -> [Float] {
        var srcBuf = src
        var dstBuf = [Float](repeating: 0, count: toW * toH)
        var srcVImage = vImage_Buffer(data: &srcBuf, height: vImagePixelCount(fromSize), width: vImagePixelCount(fromSize), rowBytes: fromSize * MemoryLayout<Float>.size)
        var dstVImage = vImage_Buffer(data: &dstBuf, height: vImagePixelCount(toH), width: vImagePixelCount(toW), rowBytes: toW * MemoryLayout<Float>.size)
        vImageScale_PlanarF(&srcVImage, &dstVImage, nil, vImage_Flags(kvImageHighQualityResampling))
        return dstBuf
    }

    /// Separable Gaussian: horizontal pass then vertical pass, edge-clamped
    /// (matches a standard bitmap Gaussian blur's border handling; the
    /// Android side does the same via its own pure-Kotlin separable
    /// implementation per the doc).
    private static func separableBlur(_ src: [Float], size: Int, kernel: [Float]) -> [Float] {
        let radius = kernel.count / 2
        var temp = [Float](repeating: 0, count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                var acc: Float = 0
                for k in -radius...radius {
                    let xs = max(0, min(size - 1, x + k))
                    acc += src[y * size + xs] * kernel[k + radius]
                }
                temp[y * size + x] = acc
            }
        }
        var out = [Float](repeating: 0, count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                var acc: Float = 0
                for k in -radius...radius {
                    let ys = max(0, min(size - 1, y + k))
                    acc += temp[ys * size + x] * kernel[k + radius]
                }
                out[y * size + x] = acc
            }
        }
        return out
    }
}
