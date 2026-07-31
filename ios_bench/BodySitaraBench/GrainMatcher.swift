import Foundation

/// Adds Gaussian film grain to a filled core region, sigma measured from
/// the real background ring immediately surrounding it -- ported from
/// Android's addGrain() (BackgroundInpaint.kt lines 703-718). Neural/push-
/// pull fills are smoother than real camera sensor noise; matching the
/// local grain keeps the fill from reading as suspiciously clean next to
/// real texture.
enum GrainMatcher {
    /// Fixed seed for determinism, matching Android's addGrain() (20260719L).
    private static let seed: UInt64 = 20_260_719

    /// Deterministic xorshift64* PRNG -- avoids a GameplayKit dependency for
    /// one Gaussian sampler, and gives the same seeded-reproducibility
    /// property as Android's `java.util.Random(20260719L)`.
    private struct SeededRNG {
        var state: UInt64
        init(seed: UInt64) { state = seed == 0 ? 0xdeadbeef : seed }
        mutating func nextUniform() -> Double {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Double(state >> 11) * (1.0 / Double(1 << 53))
        }
    }

    /// Box-Muller transform: two uniform samples -> one standard-normal sample.
    private static func nextGaussian(_ rng: inout SeededRNG) -> Double {
        let u1 = max(1e-12, rng.nextUniform())
        let u2 = rng.nextUniform()
        return sqrt(-2.0 * log(u1)) * cos(2.0 * Double.pi * u2)
    }

    static func apply(plate: RGBBuffer, core: [Bool]) -> RGBBuffer {
        let w = plate.width, h = plate.height
        let ring = BackgroundReconstructor.dilate(core, width: w, height: h, radius: 8)

        var sum = 0.0, sq = 0.0
        var count = 0
        for p in 0..<(w * h) where ring[p] && !core[p] {
            let l = (Double(plate.r[p]) + Double(plate.g[p]) + Double(plate.b[p])) / 3.0
            sum += l; sq += l * l; count += 1
        }
        guard count >= 30 else { return plate }
        let mean = sum / Double(count)
        let variance = max(0, sq / Double(count) - mean * mean)
        let sigma = min(10.0, max(1.0, sqrt(variance)))

        var rng = SeededRNG(seed: seed)
        var outR = plate.r, outG = plate.g, outB = plate.b
        for p in 0..<(w * h) where core[p] {
            let d = Float(nextGaussian(&rng) * sigma)
            outR[p] = min(255, max(0, plate.r[p] + d))
            outG[p] = min(255, max(0, plate.g[p] + d))
            outB[p] = min(255, max(0, plate.b[p] + d))
        }
        return RGBBuffer(r: outR, g: outG, b: outB, width: w, height: h)
    }
}
