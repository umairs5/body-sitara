import UIKit

/// PLACEHOLDER test-frame generator -- solid-color synthetic frames, used
/// only to get the first real on-device latency number without needing to
/// bundle real dataset clips into the app yet. Latency (pure inference
/// time) should not depend meaningfully on frame CONTENT for a
/// fixed-shape model like RIFE/LaMa, only on resolution -- so this is
/// reasonable for a first pass, but real dataset frames (matching the
/// same clips used in the Pi/Android benchmarks) should replace this
/// before reporting final comparison numbers, since content-dependent
/// early-exit or sparsity optimizations (if any exist in these models)
/// would not show up on a flat solid-color input.
enum TestFrameProvider {
    static func solidColorImage(size: Int, seed: Int, grayscale: Bool = false) -> CGImage? {
        let colorSpace = grayscale ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = grayscale ? 1 : 4
        let bitmapInfo: CGBitmapInfo = grayscale ? CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue) : CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * bytesPerPixel,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            return nil
        }

        // Deterministic per-seed color so RIFE's two frames (seed 1, 2)
        // aren't literally identical -- gives optical-flow-like ops
        // something nontrivial to compute, closer to real usage than a
        // flat single frame repeated twice.
        let r = CGFloat((seed * 37) % 256) / 255.0
        let g = CGFloat((seed * 73) % 256) / 255.0
        let b = CGFloat((seed * 111) % 256) / 255.0

        if grayscale {
            context.setFillColor(gray: r, alpha: 1.0)
        } else {
            context.setFillColor(CGColor(red: r, green: g, blue: b, alpha: 1.0))
        }
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))

        return context.makeImage()
    }
}
