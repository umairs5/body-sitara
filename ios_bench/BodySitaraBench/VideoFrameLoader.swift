import AVFoundation
import UIKit

/// Loads all frames of a real video file into memory as CGImages, for
/// feeding a real 10s dataset clip through the on-device pipeline port
/// (as opposed to TestFrameProvider's synthetic solid-color frames, which
/// only exercise raw model latency, not real clip content).
enum VideoFrameLoader {
    struct LoadedVideo {
        let frames: [CGImage]
        let width: Int
        let height: Int
    }

    static func loadFrames(url: URL, maxFrames: Int? = nil) throws -> LoadedVideo {
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

        var frames: [CGImage] = []
        let ciContext = CIContext(options: [.useSoftwareRenderer: false])

        while reader.status == .reading {
            guard let sampleBuffer = output.copyNextSampleBuffer(),
                  let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                continue
            }
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { continue }
            frames.append(cgImage)
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

        return LoadedVideo(frames: frames, width: width, height: height)
    }
}
