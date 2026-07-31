import AVFoundation
import UIKit

/// Encodes a sequence of CGImage frames into a real, viewable .mp4 file,
/// so pipeline-stage outputs (reconstructed background, composited
/// output, etc.) can be saved to Photos and scrubbed through like a
/// normal video -- not just inspected as single still frames.
///
/// STREAMING API (`StreamingVideoWriter`): this is now the ONLY way frames
/// reach an .mp4 file from this app. The previous API here was `encode
/// (frames: [CGImage], fps:, outputURL:)`, which required the CALLER to
/// first build a COMPLETE `[CGImage]` for the whole clip before this
/// function ever ran -- even though the loop inside that function only
/// ever touched ONE frame at a time (`adaptor.append(pixelBuffer,
/// withPresentationTime:)` inside a `for (i, frame) in frames.enumerated()`
/// loop; nothing about AVAssetWriter needs more than the current frame
/// resident). That pre-built-array requirement pushed the actual memory
/// cost onto three call sites in BenchmarkView.swift (the reconstructed-
/// background, silhouette-on-lightmap, and final-composite frame-
/// production loops), each of which built and held a `[CGImage]` for ALL
/// 300 frames of a real on-device test clip simultaneously so it would
/// have something to hand to `encode(frames:...)` at export time -- ~6.55MB
/// per 1280x1280 RGBA8 CGImage backing store * 300 frames = ~1.97GB per
/// array, and because each of those three arrays was still referenced
/// later (one for its own export call, a second one AGAIN inside the third
/// loop, which read `reconstructedBgFrames[i]` as its own per-frame input),
/// all three were resident AT THE SAME TIME by the point the third loop
/// finished -- ~5.9GB combined, on top of whatever the earlier (already-
/// fixed) load/align/reconstruction stages hadn't fully released yet. This
/// matched a real on-device jetsam kill that happened right after a LaMa
/// core-fill call succeeded and before the next expected log line
/// (2026-07-31).
///
/// `StreamingVideoWriter` exposes the SAME internal one-frame-at-a-time
/// shape `encode(frames:...)` always had, just as the actual public
/// surface instead of hiding it behind an array parameter: `init(...)`
/// runs the exact same `AVAssetWriter`/`AVAssetWriterInputPixelBufferAdaptor`/
/// pixel-buffer-pool setup `encode(frames:...)` used to run once per call,
/// `append(_:)` runs the exact same per-frame `CGContext` draw +
/// `adaptor.append` the old loop body ran, and `finish()` runs the exact
/// same `markAsFinished` + `finishWriting` teardown. No AVFoundation
/// behavior changed and no pixel output changed -- only which layer owns
/// the loop over frames, so callers can now discard each `CGImage`
/// immediately after `append(_:)` returns instead of accumulating an array.
///
/// MEMORY MATH for a 300-frame, 1280x1280 clip (the crash-reproducing
/// case), at this frame-production/export layer specifically -- the gap
/// the three prior fixes (aligned-frame stack, colorBuffers/maskBuffers
/// input, streaming video LOAD) didn't cover, because all three of THOSE
/// fixes addressed getting frames IN, not getting rendered stage output
/// OUT to disk:
///   Before (BenchmarkView.swift built [CGImage] per stage, encode(frames:)
///   took the whole array):
///     Each 1280x1280 RGBA8 CGImage backing store is ~6.55MB
///     (1280*1280*4 bytes). 300 of them = ~1.97GB per stage array
///     (reconstructedBgFrames, silhouetteOnLightmapFrames, finalFrames).
///     Loop 3 (Final Compositing) read `reconstructedBgFrames[i]` as its
///     own per-frame input, so Loop 1's array was STILL alive while Loop 3
///     built its own -- meaning by the time Loop 3 finished, all three
///     ~1.97GB arrays were resident simultaneously: ~5.9GB combined, on
///     top of whatever the earlier (already-fixed) load/align/
///     reconstruction stages hadn't fully released yet. This is exactly
///     where a real on-device jetsam kill happened (2026-07-31): right
///     after LaMa core-fill succeeded, i.e. right as the first of these
///     three multi-GB allocations (Loop 1) started peaking.
///   After (StreamingVideoWriter, one CGImage per iteration):
///     Each of the three stages now opens its writer, computes ONE frame,
///     calls `append(_:)`, and lets that CGImage/RGBBuffer fall out of
///     scope before computing the next -- peak resident CGImage count per
///     stage is O(1), not O(N): ~6.55MB for the current frame (plus its
///     source RGBBuffer, ~19.7MB Float32 for a single 1280x1280x3-channel
///     frame -- also O(1), never O(N)), regardless of clip length. Loop 3
///     no longer needs Loop 1's array at all -- it recomputes frame i's
///     reconstructed-background content on the fly via the shared
///     `renderReconFrame(i)` closure (see BenchmarkView.swift) instead of
///     indexing into a stored array, so Loop 1's writer can close and its
///     resources release BEFORE Loop 2 or Loop 3 even start. Net: the
///     ~5.9GB three-simultaneous-array peak collapses to roughly one
///     frame's worth of memory (~26MB: CGImage + Float32 RGBBuffer) per
///     stage, plus whatever AVAssetWriter/AVAssetWriterInputPixelBufferAdaptor
///     buffer internally for muxing -- that internal buffering is bounded
///     by the pixel-buffer-pool's small fixed pool size (a handful of
///     in-flight buffers for encoder pipelining), not by clip length, so
///     it does not reintroduce an O(N) cost.
final class StreamingVideoWriter {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let width: Int
    private let height: Int
    private let fps: Int32
    private var frameIndex: Int = 0
    private(set) var framesWritten: Int = 0

    /// Opens `outputURL` for writing and starts the AVAssetWriter session.
    /// `width`/`height` must be known up front (same requirement the old
    /// `encode(frames:...)` had implicitly, via `frames.first.width/height`)
    /// since `AVAssetWriterInput`'s output settings are fixed at
    /// construction -- every call site already knows its target clip's
    /// dimensions before its per-frame loop starts (e.g.
    /// `backgroundFinal.width/height`), so this isn't a new constraint,
    /// just made explicit instead of inferred from the first array element.
    init(outputURL: URL, width: Int, height: Int, fps: Int32) throws {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }
        self.width = width
        self.height = height
        self.fps = fps

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attrs)
        writer.add(input)

        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        self.writer = writer
        self.input = input
        self.adaptor = adaptor
    }

    /// Appends ONE frame at the next sequential presentation time.
    /// Identical per-frame work to the old `encode(frames:...)` loop body:
    /// wait for the input to be ready, pull a pixel buffer from the
    /// adaptor's pool, draw the CGImage into it via CGContext, append.
    /// `frame` is not retained by this method past the `context.draw`
    /// call -- once `append` returns, nothing here still references it, so
    /// a caller that doesn't stash it in its own array keeps peak memory
    /// at O(1) frames instead of O(N). `AVAssetWriterInput`/
    /// `AVAssetWriterInputPixelBufferAdaptor` are plain (non-actor-isolated)
    /// AVFoundation classes -- they do their own internal queuing/dispatch,
    /// nothing here requires the main actor, so this can be (and is) called
    /// from inside a `Task.detached` background block same as the rest of
    /// this file's heavy per-frame compute.
    func append(_ frame: CGImage) throws {
        while !input.isReadyForMoreMediaData {
            Thread.sleep(forTimeInterval: 0.005)
        }
        guard let pixelBufferPool = adaptor.pixelBufferPool else {
            throw NSError(domain: "VideoEncoder", code: 2, userInfo: [NSLocalizedDescriptionKey: "no pixel buffer pool"])
        }
        var pixelBufferOut: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &pixelBufferOut)
        guard let pixelBuffer = pixelBufferOut else { return }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { return }
        context.draw(frame, in: CGRect(x: 0, y: 0, width: width, height: height))

        let presentationTime = CMTime(value: CMTimeValue(frameIndex), timescale: fps)
        adaptor.append(pixelBuffer, withPresentationTime: presentationTime)
        frameIndex += 1
        framesWritten += 1
    }

    /// Closes out the AVAssetWriter session -- identical teardown to the
    /// old `encode(frames:...)`'s tail (`markAsFinished` + `finishWriting`
    /// + status check), just expressed with `async`/`await` instead of a
    /// blocking `DispatchSemaphore.wait()` since every call site is already
    /// inside an `async` context.
    func finish() async throws {
        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }
        if writer.status == .failed {
            throw writer.error ?? NSError(domain: "VideoEncoder", code: 3, userInfo: [NSLocalizedDescriptionKey: "unknown write failure"])
        }
    }
}

enum VideoEncoder {
    /// Saves a video file at `url` to the user's Photos library. Requires
    /// NSPhotoLibraryAddUsageDescription in Info.plist. Unaffected by the
    /// streaming rework above -- it only ever took a finished file's URL,
    /// never frame data.
    static func saveToPhotoLibrary(url: URL) async throws {
        try await PHPhotoLibrarySaver.save(videoURL: url)
    }
}

import Photos

private enum PHPhotoLibrarySaver {
    static func save(videoURL: URL) async throws {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            throw NSError(domain: "VideoEncoder", code: 4, userInfo: [NSLocalizedDescriptionKey: "Photos permission denied (\(status.rawValue))"])
        }
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: videoURL)
        }
    }
}
