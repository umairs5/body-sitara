import AVFoundation
import UIKit

/// Encodes a sequence of CGImage frames into a real, viewable .mp4 file,
/// so pipeline-stage outputs (reconstructed background, composited
/// output, etc.) can be saved to Photos and scrubbed through like a
/// normal video -- not just inspected as single still frames.
enum VideoEncoder {
    static func encode(frames: [CGImage], fps: Int32, outputURL: URL) throws {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }
        guard let first = frames.first else {
            throw NSError(domain: "VideoEncoder", code: 1, userInfo: [NSLocalizedDescriptionKey: "no frames to encode"])
        }
        let width = first.width
        let height = first.height

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

        for (i, frame) in frames.enumerated() {
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.005)
            }
            guard let pixelBufferPool = adaptor.pixelBufferPool else {
                throw NSError(domain: "VideoEncoder", code: 2, userInfo: [NSLocalizedDescriptionKey: "no pixel buffer pool"])
            }
            var pixelBufferOut: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &pixelBufferOut)
            guard let pixelBuffer = pixelBufferOut else { continue }

            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
            guard let context = CGContext(
                data: CVPixelBufferGetBaseAddress(pixelBuffer),
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
            ) else { continue }
            context.draw(frame, in: CGRect(x: 0, y: 0, width: width, height: height))

            let presentationTime = CMTime(value: CMTimeValue(i), timescale: fps)
            adaptor.append(pixelBuffer, withPresentationTime: presentationTime)
        }

        input.markAsFinished()
        let sema = DispatchSemaphore(value: 0)
        writer.finishWriting { sema.signal() }
        sema.wait()

        if writer.status == .failed {
            throw writer.error ?? NSError(domain: "VideoEncoder", code: 3, userInfo: [NSLocalizedDescriptionKey: "unknown write failure"])
        }
    }

    /// Saves a video file at `url` to the user's Photos library. Requires
    /// NSPhotoLibraryAddUsageDescription in Info.plist.
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
