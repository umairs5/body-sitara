import SwiftUI

/// Fullscreen before/after: the masked input clip stacked above the
/// finished pipeline output, both looping simultaneously -- the fastest
/// way to eyeball "did this run actually work" without scrubbing two
/// separate exports in Photos. Modeled on Danial's Android Compare card
/// (MainActivity.kt bindCollage/openCollagePlayer's bottom "masked vs
/// final" pairing), adapted to a two-pane SwiftUI stack.
struct CompareView: View {
    let beforeTitle: String
    let beforeURL: URL
    let afterTitle: String
    let afterURL: URL
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            paneHeader(beforeTitle, tint: Theme.accent)
            LoopingVideoPlayer(url: beforeURL)
                .frame(maxHeight: .infinity)

            paneHeader(afterTitle, tint: Theme.success)
            LoopingVideoPlayer(url: afterURL)
                .frame(maxHeight: .infinity)
        }
        .background(Color.black)
        .ignoresSafeArea(edges: .bottom)
        .overlay(alignment: .topTrailing) {
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title)
                    .foregroundStyle(.white, .black.opacity(0.5))
                    .padding()
            }
        }
    }

    private func paneHeader(_ text: String, tint: Color) -> some View {
        HStack {
            Circle().fill(tint).frame(width: 8, height: 8)
            Text(text)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.black.opacity(0.8))
    }
}
