import SwiftUI

enum StageStatus: Equatable {
    case pending
    case running
    case done
    case failed(String)
}

/// One row of the pipeline timeline (e.g. "Background Reconstruction",
/// "Illumination Extraction"). `timings` are labeled sub-costs shown as
/// chips (e.g. "align" / "trim" / "fill"); `methodBadge` surfaces which
/// strategy actually ran where that matters (LaMa bbox-crop vs push-pull
/// anti-hallucination fallback) so a glance at the dashboard answers "did
/// the near-static guard trigger?" without reading the log.
struct PipelineStage: Identifiable {
    let id = UUID()
    var name: String
    var icon: String
    var status: StageStatus = .pending
    var timings: [(label: String, ms: Double)] = []
    var methodBadge: (text: String, isWarning: Bool)? = nil
    var detailLines: [String] = []
}

struct PipelineStageView: View {
    let stage: PipelineStage
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                statusIcon
                Text(stage.name)
                    .font(.subheadline.weight(.medium))
                Spacer()
                if let badge = stage.methodBadge {
                    Badge(text: badge.text, color: badge.isWarning ? Theme.warning : Theme.success)
                }
                if !stage.timings.isEmpty {
                    HStack(spacing: 10) {
                        ForEach(stage.timings, id: \.label) { t in
                            TimingChip(label: t.label, ms: t.ms)
                        }
                    }
                }
                if !stage.detailLines.isEmpty {
                    Button {
                        withAnimation(.snappy) { isExpanded.toggle() }
                    } label: {
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }

            if case .failed(let message) = stage.status {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(Theme.danger)
            }

            if isExpanded && !stage.detailLines.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(stage.detailLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.leading, 28)
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch stage.status {
        case .pending:
            Image(systemName: "circle.dotted").foregroundStyle(.secondary)
        case .running:
            ProgressView().controlSize(.small)
        case .done:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.success)
        case .failed:
            Image(systemName: "xmark.circle.fill").foregroundStyle(Theme.danger)
        }
    }
}

/// The full vertical timeline card.
struct PipelineTimelineView: View {
    let stages: [PipelineStage]

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 4) {
                SectionHeader(icon: "point.3.filled.connected.trianglepath.dotted", title: "Pipeline")
                Divider().padding(.vertical, 4)
                ForEach(Array(stages.enumerated()), id: \.element.id) { index, stage in
                    PipelineStageView(stage: stage)
                    if index < stages.count - 1 {
                        Divider()
                    }
                }
            }
        }
    }
}
