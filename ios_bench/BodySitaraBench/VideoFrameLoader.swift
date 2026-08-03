import AVFoundation
import UIKit

/// Streams frames of a real video file into memory, converting each decoded
/// `CGImage` to its compact per-frame representation (`RGBBuffer8`,
/// `PackedMaskFrame`, or `GrayBuffer8`) INSIDE the decode loop, one frame at
/// a time.
///
/// This replaces an earlier design that returned `[CGImage]` for the WHOLE
/// clip (`LoadedVideo(frames: [CGImage], ...)`) and left conversion to the
/// caller. That earlier design was a THIRD, previously-unaddressed memory
/// sink -- distinct from (and upstream of) the two already-fixed sinks in
/// `BackgroundReconstructor.swift` (internal aligned-frame stack) and
/// `PixelBuffer.swift`'s `RGBBuffer8`/`PackedMaskFrame` (the colorBuffers/
/// maskBuffers input array). On a real device, a 300-frame/1280x1280 clip
/// crashed (jetsam/OOM kill) almost immediately, during "Load & Align"
/// itself, BEFORE Background Reconstruction (where the previous two fixes'
/// bounded code runs) ever started -- because `loadFrames` fully decoded
/// video 1 into 300 resident `CGImage`s, THEN fully decoded video 2 into
/// another 300 resident `CGImage`s while video 1's were still held (both
/// `maskedVideo`/`maskVideo` locals stay in scope for the rest of the
/// calling function, including several `addPreview` calls that reference
/// `.frames[...]` after both loads complete). See the MEMORY MATH doc below
/// for the actual before/after numbers.
///
/// MEMORY MATH for a 300-frame, 1280x1280 clip (the crash-reproducing case),
/// at THIS decode/load layer specifically:
///   Before (loadFrames returns [CGImage], full clip resident):
///     Each 1280x1280 RGBA8 CGImage backing store is ~6.55MB
///     (1280*1280*4 bytes). 300 of them = ~1.97GB PER VIDEO. With both
///     `maskedVideo.frames` and `maskVideo.frames` alive simultaneously
///     (the addPreview calls need both after both loads finish), that's
///     ~3.9GB of CGImage backing stores ALONE -- on top of whatever
///     CIContext/Metal-side retains per createCGImage call, and BEFORE the
///     already-fixed ~1.47GB colorBuffers/~60MB maskBuffers conversion step
///     even begins. This is large enough to jetsam-kill on its own, and
///     matches the observed "crash during Load & Align, not Background
///     Reconstruction" symptom.
///   After (loadFramesAsRGBBuffer8 / loadFramesAsPackedMask, streaming):
///     At most ONE CGImage is resident at any instant during the decode
///     loop (the current sample buffer's), immediately converted to
///     RGBBuffer8 (3 bytes/pixel) or PackedMaskFrame (1 bit/pixel) and the
///     CGImage dropped before the next `copyNextSampleBuffer()` call. Peak
///     resident set for the LOADING pass itself is therefore O(1) frames of
///     CGImage overhead (~6.55MB, immediately reused/reclaimed by ARC each
///     iteration) plus the compact array being built in place -- i.e. the
///     loading pass no longer adds its own multi-GB peak on top of the
///     already-fixed ~1.47GB colorBuffers + ~60MB maskBuffers; it now
///     costs roughly what those already cost, not ~4GB more.
///   Two preview CGImages (frame 0 and the mid-clip frame of the masked
///   video) are captured directly during this same streaming pass -- they
///   are 2 single frames (~13MB total), not a problem to keep as CGImage,
///   and this avoids reconstructing them from RGBBuffer8 afterward.
enum VideoFrameLoader {
    /// Result of streaming-decoding the MASKED (color) video: the compact
    /// per-frame color buffers plus the two preview `CGImage`s the caller's
    /// gallery needs (frame 0 and the mid-clip frame -- the mid-clip index
    /// depends on the OTHER video's frame count too via `n = min(...)`, so
    /// the caller passes which extra indices to snapshot).
    struct LoadedColorVideo {
        let frames: [RGBBuffer8]
        let width: Int
        let height: Int
        /// CGImage previews captured at the requested `previewIndices`,
        /// keyed by that index. Empty entries (index out of range) are
        /// simply absent.
        let previews: [Int: CGImage]
    }

    struct LoadedMaskVideo {
        let frames: [PackedMaskFrame]
        let width: Int
        let height: Int
        let previews: [Int: CGImage]
    }

    /// Result of streaming-decoding a single-channel (grayscale-in-RGB)
    /// video, e.g. a cloud-rendered alpha matte -- see `GrayBuffer8`'s doc
    /// comment in PixelBuffer.swift for why this exists as a distinct,
    /// non-bit-packed 1-byte/pixel type rather than reusing `RGBBuffer8`
    /// (which the alpha matte was loaded through previously) or
    /// `PackedMaskFrame` (which would hard-clip its soft, antialiased
    /// edges).
    struct LoadedGrayVideo {
        let frames: [GrayBuffer8]
        let width: Int
        let height: Int
        let previews: [Int: CGImage]
    }

    /// Streams the masked/color video, converting each frame to `RGBBuffer8`
    /// as it's decoded so at most one `CGImage` is resident at a time.
    /// `previewIndices` lets the caller grab a small number of frames as
    /// actual `CGImage`s (for UI preview galleries) without materializing
    /// the whole clip as `CGImage`s -- those specific frames are cloned into
    /// `previews` at the moment they're decoded, alongside the normal
    /// RGBBuffer8 conversion.
    /// Also reused, unmodified, to load the real synthetic-CHARACTER video
    /// in BenchmarkView's Final Compositing stage, when a `ClipPreset` has
    /// `hasCharacter == true` -- just another color clip, same
    /// decode/streaming requirements as masked_video.mp4. (The character's
    /// ALPHA MATTE video is loaded via `loadFramesAsGrayBuffer8` below
    /// instead, as of the 2026-08-03 memory-budget pass -- see that
    /// function's doc comment.)
    static func loadFramesAsRGBBuffer8(url: URL, previewIndices: Set<Int> = [], maxFrames: Int? = nil) throws -> LoadedColorVideo {
        let (frames, previews, width, height) = try streamFrames(url: url, previewIndices: previewIndices, maxFrames: maxFrames) { cgImage in
            RGBBuffer8.from(cgImage: cgImage)
        }
        return LoadedColorVideo(frames: frames, width: width, height: height, previews: previews)
    }

    /// Streams the mask video, converting each frame to a bit-packed
    /// `PackedMaskFrame` as it's decoded. Same one-CGImage-at-a-time
    /// discipline as `loadFramesAsRGBBuffer8`.
    static func loadFramesAsPackedMask(url: URL, previewIndices: Set<Int> = [], maxFrames: Int? = nil) throws -> LoadedMaskVideo {
        let (frames, previews, width, height) = try streamFrames(url: url, previewIndices: previewIndices, maxFrames: maxFrames) { cgImage in
            PackedMaskFrame.from(cgImage: cgImage)
        }
        return LoadedMaskVideo(frames: frames, width: width, height: height, previews: previews)
    }

    /// Streams a single-channel (grayscale-in-RGB) video, converting each
    /// frame to a `GrayBuffer8` as it's decoded -- same one-CGImage-at-a-time
    /// discipline as `loadFramesAsRGBBuffer8`/`loadFramesAsPackedMask`.
    ///
    /// Added for the character's ALPHA MATTE video specifically (real
    /// synthetic-character compositing, memory-budget review 2026-08-03):
    /// it used to be loaded through `loadFramesAsRGBBuffer8` purely to reuse
    /// this same streaming decode loop, which meant paying for 3 redundant
    /// bytes/pixel (R==G==B) when only 1 channel's worth of information is
    /// ever read back. `loadFramesAsPackedMask` was NOT the right fix
    /// instead -- it's 1-bit-per-pixel, correct for a strictly binary
    /// segmentation mask but wrong for a soft, antialiased alpha matte (see
    /// the ALPHA-MATTE PRECISION DECISION doc on `RGBBuffer8` in
    /// PixelBuffer.swift). `GrayBuffer8` keeps the same full 0-255
    /// precision as before, just without the two unused channels: 1
    /// byte/pixel instead of 3, a lossless 3x reduction (~1.44GB -> ~0.48GB
    /// at 300 frames/1264x1264).
    static func loadFramesAsGrayBuffer8(url: URL, previewIndices: Set<Int> = [], maxFrames: Int? = nil) throws -> LoadedGrayVideo {
        let (frames, previews, width, height) = try streamFrames(url: url, previewIndices: previewIndices, maxFrames: maxFrames) { cgImage in
            GrayBuffer8.from(cgImage: cgImage)
        }
        return LoadedGrayVideo(frames: frames, width: width, height: height, previews: previews)
    }

    /// Shared decode loop: identical `AVAssetReader`/`CIContext` setup and
    /// read loop to the previous `loadFrames` implementation (deliberately
    /// unchanged -- see header doc), except the per-frame `CGImage` is
    /// converted via `convert` and discarded immediately instead of being
    /// appended to a clip-wide `[CGImage]` array. Frames whose index is in
    /// `previewIndices` are additionally kept as `CGImage` (that's the ONLY
    /// case where a decoded `CGImage` outlives its loop iteration).
    private static func streamFrames<T>(
        url: URL,
        previewIndices: Set<Int>,
        maxFrames: Int?,
        convert: (CGImage) -> T
    ) throws -> (frames: [T], previews: [Int: CGImage], width: Int, height: Int) {
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw NSError(domain: "VideoFrameLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "no video track in \(url.lastPathComponent)"])
        }

        let reader = try AVAssetReader(asset: asset)
        let outputSettings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        reader.add(output)
        reader.startReading()

        let naturalSize = track.naturalSize
        let width = Int(abs(naturalSize.width))
        let height = Int(abs(naturalSize.height))

        // Same CIContext usage as before this fix -- left untouched per the
        // task's guidance not to introduce a second variable (renderer/
        // decode path change) alongside the accumulation fix, so an
        // on-device retest isolates whether THIS fix (no more [CGImage]
        // accumulation) is what worked. Each `cgImage` below is a local
        // `let` that falls out of scope at the end of its loop iteration
        // (or right after being cloned into `previews`), which is enough
        // for ARC to release its CPU-side backing store and let the
        // CIContext/Metal-side texture cache reclaim GPU-side resources
        // between frames -- there is no separate "reset" API needed for
        // that on CIContext.
        var frames: [T] = []
        var previews: [Int: CGImage] = [:]
        var index = 0
        let ciContext = CIContext(options: [.useSoftwareRenderer: false])

        while reader.status == .reading {
            guard let sampleBuffer = output.copyNextSampleBuffer(),
                  let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                continue
            }
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { continue }

            if previewIndices.contains(index) {
                previews[index] = cgImage
            }
            frames.append(convert(cgImage))
            index += 1

            if let maxFrames, frames.count >= maxFrames {
                reader.cancelReading()
                break
            }
        }

        guard reader.status == .completed || reader.status == .cancelled else {
            throw NSError(domain: "VideoFrameLoader", code: 2, userInfo: [NSLocalizedDescriptionKey: "AVAssetReader failed: \(reader.error?.localizedDescription ?? "unknown")"])
        }
        guard !frames.isEmpty else {
            throw NSError(domain: "VideoFrameLoader", code: 3, userInfo: [NSLocalizedDescriptionKey: "no frames decoded from \(url.lastPathComponent)"])
        }

        return (frames, previews, width, height)
    }
}
