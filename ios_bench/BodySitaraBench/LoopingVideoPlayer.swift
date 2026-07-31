import AVKit
import SwiftUI

/// A looping AVKit video player -- used both for the fullscreen clip
/// viewer and (at small size) for inline playing thumbnails, so staged
/// input clips and exported pipeline outputs can actually be WATCHED
/// in-app, not just inspected as a single still frame or found later in
/// Photos. Loops via AVPlayerLooper + AVQueuePlayer (the standard non-
/// flickering loop mechanism, rather than reseeking to zero on
/// AVPlayerItemDidPlayToEndTime, which visibly stutters).
struct LoopingVideoPlayer: View {
    let url: URL
    @State private var queuePlayer: AVQueuePlayer?
    @State private var looper: AVPlayerLooper?

    var body: some View {
        Group {
            if let queuePlayer {
                VideoPlayer(player: queuePlayer)
                    .onAppear { queuePlayer.play() }
                    .onDisappear { queuePlayer.pause() }
            } else {
                Color.black
            }
        }
        .onAppear(perform: setUp)
        .onChange(of: url) { _, _ in setUp() }
    }

    private func setUp() {
        let item = AVPlayerItem(url: url)
        let player = AVQueuePlayer()
        looper = AVPlayerLooper(player: player, templateItem: item)
        queuePlayer = player
        player.play()
    }
}

/// Fullscreen sheet wrapper: label header + looping player + a close
/// affordance (tap anywhere, matching Danial's "tap anywhere to close"
/// collage-player convention).
struct VideoViewerSheet: View {
    let title: String
    let url: URL
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()
            LoopingVideoPlayer(url: url)
                .ignoresSafeArea()
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title)
                    .foregroundStyle(.white, .black.opacity(0.5))
                    .padding()
            }
        }
        .overlay(alignment: .top) {
            Text(title)
                .font(.headline)
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.black.opacity(0.5), in: Capsule())
                .padding(.top, 50)
        }
    }
}
