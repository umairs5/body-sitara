import SwiftUI

/// Small shared visual system for the benchmark dashboard -- semantic
/// colors and a couple of reusable container styles, so every card/badge
/// in BenchmarkView, ClipPickerView, PipelineStageView, and CompareView
/// reads as one consistent app instead of ad hoc styling per view. Uses
/// system colors throughout (never hardcoded RGB) so it stays correct in
/// both light and dark mode automatically.
enum Theme {
    static let accent = Color.accentColor
    static let success = Color.green
    static let warning = Color.orange
    static let danger = Color.red
    static let cardBackground = Color(uiColor: .secondarySystemGroupedBackground)
    static let pageBackground = Color(uiColor: .systemGroupedBackground)

    static let cardCornerRadius: CGFloat = 16
    static let smallCornerRadius: CGFloat = 10
}

/// Standard card container: rounded, subtly elevated, consistent padding.
/// Every top-level section in the dashboard (Clips, Run Control, Pipeline
/// Timeline, Gallery, Log) sits inside one of these.
struct Card<Content: View>: View {
    let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }

    var body: some View {
        content
            .padding(16)
            .background(Theme.cardBackground, in: RoundedRectangle(cornerRadius: Theme.cardCornerRadius, style: .continuous))
    }
}

/// Section header used above each card: icon + title, consistent weight/
/// size across the dashboard.
struct SectionHeader: View {
    let icon: String
    let title: String
    var trailing: AnyView? = nil

    var body: some View {
        HStack {
            Label(title, systemImage: icon)
                .font(.headline)
            Spacer()
            if let trailing { trailing }
        }
    }
}

/// Small rounded status/method badge -- used for stage status, core-fill
/// method (lama-crop vs push-pull), and clip-preset completeness.
struct Badge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}

/// A single timing number rendered as a small chip, e.g. "142 ms".
struct TimingChip: View {
    let label: String
    let ms: Double

    var body: some View {
        VStack(spacing: 2) {
            Text(String(format: "%.0f", ms))
                .font(.system(.subheadline, design: .rounded).weight(.semibold))
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 52)
    }
}
