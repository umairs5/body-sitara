import CoreGraphics

/// Swift port of Danial's Compositor/NCompositor + RelightPhase (Android
/// phases 2/2b, per Tier_2_Mobile_Comprehensive_Guide.md). Scope note:
/// uses a PLACEHOLDER synthetic-character cutout (solid-color RGBA with a
/// soft alpha edge, generated locally) rather than a real cloud-returned
/// WanAnimate render, per explicit instruction -- this tests the
/// on-device compositing/relight MATH cost, not end-to-end visual
/// fidelity, which needs a real character asset to evaluate separately.
enum Compositor {
    struct Result {
        let composited: RGBBuffer
        let relit: RGBBuffer
        let compositeMs: Double
        let relightMs: Double
    }

    struct CompositeOnlyResult {
        let composited: RGBBuffer
        let compositeMs: Double
    }

    /// Alpha composite ONLY, no relight -- for the Tier2-mobile system-cost
    /// benchmark, per explicit scope decision to exclude relight timing/
    /// output from that report while its correctness is still being
    /// verified independently of Background Reconstruction's alignment
    /// bug (see BackgroundReconstructor.swift's warpTranslate fix note).
    static func compositeOnly(background: RGBBuffer, character: RGBBuffer, alpha: [Float]) -> CompositeOnlyResult {
        let w = background.width, h = background.height
        precondition(character.width == w && character.height == h, "character must match background size")

        let tStart = CFAbsoluteTimeGetCurrent()
        var compR = [Float](repeating: 0, count: w * h)
        var compG = [Float](repeating: 0, count: w * h)
        var compB = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let a = alpha[i]
            compR[i] = character.r[i] * a + background.r[i] * (1 - a)
            compG[i] = character.g[i] * a + background.g[i] * (1 - a)
            compB[i] = character.b[i] * a + background.b[i] * (1 - a)
        }
        let compositeMs = (CFAbsoluteTimeGetCurrent() - tStart) * 1000
        return CompositeOnlyResult(composited: RGBBuffer(r: compR, g: compG, b: compB, width: w, height: h), compositeMs: compositeMs)
    }

    /// One-shot single-character alpha composite + relight over a
    /// reconstructed background. `character` and `alpha` (0-1, same size
    /// as background) represent the placeholder cutout; `lightmap` is
    /// LightmapExtractor's output.
    static func compositeAndRelight(background: RGBBuffer, character: RGBBuffer, alpha: [Float], lightmap: RGBBuffer) -> Result {
        let w = background.width, h = background.height
        precondition(character.width == w && character.height == h, "character must match background size")

        let tCompositeStart = CFAbsoluteTimeGetCurrent()
        var compR = [Float](repeating: 0, count: w * h)
        var compG = [Float](repeating: 0, count: w * h)
        var compB = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let a = alpha[i]
            compR[i] = character.r[i] * a + background.r[i] * (1 - a)
            compG[i] = character.g[i] * a + background.g[i] * (1 - a)
            compB[i] = character.b[i] * a + background.b[i] * (1 - a)
        }
        let compositeMs = (CFAbsoluteTimeGetCurrent() - tCompositeStart) * 1000
        let composited = RGBBuffer(r: compR, g: compG, b: compB, width: w, height: h)

        let tRelightStart = CFAbsoluteTimeGetCurrent()
        let relit = relight(composite: composited, background: background, lightmap: lightmap, alpha: alpha)
        let relightMs = (CFAbsoluteTimeGetCurrent() - tRelightStart) * 1000

        return Result(composited: composited, relit: relit, compositeMs: compositeMs, relightMs: relightMs)
    }

    /// BT.709 YCbCr luma-gain harmonization, matching RelightPhase.kt: gain
    /// = targetY / charMeanY with a dead-zone (|ratio-1|<0.04 -> no-op),
    /// asymmetric response (linear headroom-scaled when brightening, gamma
    /// 0.8 when darkening), and shadow-gated ambient chroma tint via
    /// smoothstep(0.25, 0.60, Y/255) -- applied only inside the character's
    /// alpha region, matching the per-pixel scope of the Android version.
    private static func relight(composite: RGBBuffer, background: RGBBuffer, lightmap: RGBBuffer, alpha: [Float]) -> RGBBuffer {
        let w = composite.width, h = composite.height
        var outR = composite.r
        var outG = composite.g
        var outB = composite.b

        // Character mean luma (BT.709), restricted to alpha>0.5 pixels --
        // matches "charMeanY" in RelightPhase.kt.
        var sumY: Float = 0
        var count: Float = 0
        for i in 0..<(w * h) where alpha[i] > 0.5 {
            let y709 = 0.2126 * composite.r[i] + 0.7152 * composite.g[i] + 0.0722 * composite.b[i]
            sumY += y709
            count += 1
        }
        guard count > 0 else { return composite }
        let charMeanY = sumY / count

        for i in 0..<(w * h) where alpha[i] > 0.5 {
            let targetY = 0.2126 * lightmap.r[i] + 0.7152 * lightmap.g[i] + 0.0722 * lightmap.b[i]
            let ratio = targetY / max(1.0, charMeanY)

            let r0 = composite.r[i], g0 = composite.g[i], b0 = composite.b[i]
            let y0 = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0
            let cb = -0.1146 * r0 - 0.3854 * g0 + 0.5 * b0
            let cr = 0.5 * r0 - 0.4542 * g0 - 0.0458 * b0

            var y1 = y0
            if abs(ratio - 1.0) >= 0.04 {
                if ratio > 1.0 {
                    let headroom = (255.0 - y0) / 255.0
                    y1 = y0 + (targetY - y0) * headroom
                } else {
                    y1 = y0 * pow(ratio, 0.8)
                }
            }

            // Ambient chroma tint gated to shadows via smoothstep(0.25, 0.60, Y/255).
            let t = smoothstep(0.25, 0.60, y0 / 255.0)
            let ambientCb = -0.1146 * background.r[i] - 0.3854 * background.g[i] + 0.5 * background.b[i]
            let ambientCr = 0.5 * background.r[i] - 0.4542 * background.g[i] - 0.0458 * background.b[i]
            let cb1 = cb + (ambientCb - cb) * (1 - t) * 0.15
            let cr1 = cr + (ambientCr - cr) * (1 - t) * 0.15

            let r1 = y1 + 1.5748 * cr1
            let g1 = y1 - 0.1873 * cb1 - 0.4681 * cr1
            let b1 = y1 + 1.8556 * cb1

            let a = alpha[i]
            outR[i] = max(0, min(255, r1)) * a + background.r[i] * (1 - a)
            outG[i] = max(0, min(255, g1)) * a + background.g[i] * (1 - a)
            outB[i] = max(0, min(255, b1)) * a + background.b[i] * (1 - a)
        }

        return RGBBuffer(r: outR, g: outG, b: outB, width: w, height: h)
    }

    private static func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        let t = max(0, min(1, (x - edge0) / (edge1 - edge0)))
        return t * t * (3 - 2 * t)
    }

    /// Placeholder character: a soft-edged ellipse cutout (person-shaped
    /// silhouette proxy), solid mid-gray fill -- stands in for a real
    /// WanAnimate render per explicit scope decision, just to exercise the
    /// composite+relight math cost on a real background size.
    static func placeholderCharacter(width: Int, height: Int) -> (character: RGBBuffer, alpha: [Float]) {
        placeholderCharacter(width: width, height: height, frameIndex: 0, totalFrames: 1)
    }

    /// Per-frame placeholder character: the same soft-edged ellipse cutout
    /// as `placeholderCharacter(width:height:)`, but with a small per-frame
    /// horizontal sway and a breathing vertical-radius pulse driven by
    /// `frameIndex`/`totalFrames` -- so a per-frame Final Compositing loop
    /// (BenchmarkView) actually recomputes a DIFFERENT character each call,
    /// like Android's real WanAnimate output would (a moving generated
    /// character, not one static frame reused N times). This does NOT
    /// stand in for real character content/appearance -- it exists solely
    /// so Compositing's timing is measured against realistic PER-FRAME
    /// data volume/variation, not a single cached alpha mask. See
    /// BenchmarkView's Final Compositing stage for how this is invoked.
    static func placeholderCharacter(width: Int, height: Int, frameIndex: Int, totalFrames: Int) -> (character: RGBBuffer, alpha: [Float]) {
        var r = [Float](repeating: 160, count: width * height)
        var g = [Float](repeating: 140, count: width * height)
        var b = [Float](repeating: 120, count: width * height)
        var alpha = [Float](repeating: 0, count: width * height)

        // Slow horizontal sway (+-6% of width) and a gentle vertical-radius
        // breathing pulse (+-4%), phased over the clip so consecutive
        // frames differ but the motion stays smooth -- enough per-frame
        // variation that compositeOnly() can't shortcut on identical input,
        // without pretending to be a real character animation.
        let t = totalFrames > 1 ? Double(frameIndex) / Double(totalFrames - 1) : 0.0
        let sway = sin(t * 2.0 * .pi) * Double(width) * 0.06
        let breathe = 1.0 + 0.04 * sin(t * 4.0 * .pi)

        let cx = Double(width) / 2 + sway, cy = Double(height) / 2
        let rx = Double(width) * 0.18, ry = Double(height) * 0.38 * breathe
        let featherPx = 6.0

        for y in 0..<height {
            for x in 0..<width {
                let dx = (Double(x) - cx) / rx
                let dy = (Double(y) - cy) / ry
                let dist = sqrt(dx * dx + dy * dy)
                let a: Float
                if dist <= 1.0 {
                    a = 1.0
                } else if dist <= 1.0 + featherPx / rx {
                    a = Float(max(0, 1.0 - (dist - 1.0) / (featherPx / rx)))
                } else {
                    a = 0
                }
                let idx = y * width + x
                alpha[idx] = a
                if a == 0 { r[idx] = 0; g[idx] = 0; b[idx] = 0 }
            }
        }
        return (RGBBuffer(r: r, g: g, b: b, width: width, height: height), alpha)
    }
}
