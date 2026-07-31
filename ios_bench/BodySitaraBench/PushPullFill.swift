import Foundation

/// Structureless multi-resolution "push-pull" diffusion fill, ported from
/// Android's BackgroundInpaint.kt pushPull() (lines 869-917). Builds a
/// weighted mip pyramid down to 1x1 (push), then upsamples back with
/// bilinear blending toward full weight (pull), filling `hole` pixels of
/// `plate` with smooth gradients extrapolated from the surrounding real
/// pixels.
///
/// This can never invent a person-shaped hallucination the way a neural
/// inpainter can, so it's Android's fallback whenever a hole is too large
/// to trust to LaMa (see BackgroundReconstructor's anti-hallucination
/// guard, mirroring fillCore()'s `corePx > np * 0.35` branch) or whenever
/// no LaMa model is available at all.
enum PushPullFill {
    /// Fills `hole` pixels of `plate` (RGBBuffer) in place, returning a new
    /// buffer with the fill applied. `hole` pixels elsewhere are untouched.
    static func fill(plate: RGBBuffer, hole: [Bool]) -> RGBBuffer {
        let w = plate.width, h = plate.height

        var colorLevels: [[Float]] = []   // each level: interleaved RGB, count = levelW*levelH*3
        var weightLevels: [[Float]] = []
        var dims: [(w: Int, h: Int)] = []

        var c0 = [Float](repeating: 0, count: w * h * 3)
        var w0 = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) where !hole[i] {
            c0[3 * i] = plate.r[i]; c0[3 * i + 1] = plate.g[i]; c0[3 * i + 2] = plate.b[i]
            w0[i] = 1
        }
        colorLevels.append(c0); weightLevels.append(w0); dims.append((w, h))

        // Push: build the pyramid down to 1x1, averaging 2x2 blocks
        // weighted by how much real data each child pixel carries.
        var lw = w, lh = h
        while lw > 1 || lh > 1 {
            let nw = (lw + 1) / 2, nh = (lh + 1) / 2
            var nc = [Float](repeating: 0, count: nw * nh * 3)
            var nwgt = [Float](repeating: 0, count: nw * nh)
            let pc = colorLevels.last!, pw = weightLevels.last!
            let pW = lw, pH = lh
            for y in 0..<nh {
                for x in 0..<nw {
                    var r: Float = 0, g: Float = 0, b: Float = 0, wsum: Float = 0
                    for dy in 0..<2 {
                        for dx in 0..<2 {
                            let px = x * 2 + dx, py = y * 2 + dy
                            guard px < pW, py < pH else { continue }
                            let pi = py * pW + px
                            let ww = pw[pi]
                            r += pc[3 * pi] * ww; g += pc[3 * pi + 1] * ww; b += pc[3 * pi + 2] * ww
                            wsum += ww
                        }
                    }
                    let ni = y * nw + x
                    if wsum > 0 {
                        nc[3 * ni] = r / wsum; nc[3 * ni + 1] = g / wsum; nc[3 * ni + 2] = b / wsum
                        nwgt[ni] = min(1, wsum / 4)
                    }
                }
            }
            colorLevels.append(nc); weightLevels.append(nwgt); dims.append((nw, nh))
            lw = nw; lh = nh
        }

        // Pull: upsample from the coarsest level back down, bilinear-
        // blending each level's own (weak) data with the coarser level's
        // (denser) estimate, in proportion to how little real data this
        // level's pixel has (weight < 1).
        for l in stride(from: colorLevels.count - 2, through: 0, by: -1) {
            var cc = colorLevels[l]
            let cwt = weightLevels[l]
            let (cw, ch) = dims[l]
            let fc = colorLevels[l + 1]
            let (fw, fh) = dims[l + 1]
            for y in 0..<ch {
                for x in 0..<cw {
                    let i = y * cw + x
                    guard cwt[i] < 1 else { continue }
                    let a = cwt[i]
                    let fx = Float(x) * 0.5, fy = Float(y) * 0.5
                    let x0 = max(0, min(fw - 1, Int(fx))), y0 = max(0, min(fh - 1, Int(fy)))
                    let x1 = min(fw - 1, x0 + 1), y1 = min(fh - 1, y0 + 1)
                    let tx = fx - Float(x0), ty = fy - Float(y0)
                    for k in 0..<3 {
                        let c00 = fc[3 * (y0 * fw + x0) + k], c10 = fc[3 * (y0 * fw + x1) + k]
                        let c01 = fc[3 * (y1 * fw + x0) + k], c11 = fc[3 * (y1 * fw + x1) + k]
                        let v = (c00 * (1 - tx) + c10 * tx) * (1 - ty) + (c01 * (1 - tx) + c11 * tx) * ty
                        cc[3 * i + k] = a * cc[3 * i + k] + (1 - a) * v
                    }
                }
            }
            colorLevels[l] = cc
        }

        let fin = colorLevels[0]
        var outR = plate.r, outG = plate.g, outB = plate.b
        for i in 0..<(w * h) where hole[i] {
            outR[i] = min(255, max(0, fin[3 * i]))
            outG[i] = min(255, max(0, fin[3 * i + 1]))
            outB[i] = min(255, max(0, fin[3 * i + 2]))
        }
        return RGBBuffer(r: outR, g: outG, b: outB, width: w, height: h)
    }
}
