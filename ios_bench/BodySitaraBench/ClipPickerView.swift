import PhotosUI
import SwiftUI

/// Clips card: lists saved clip presets, lets you create a new one (name +
/// stage a masked-video + mask via PhotosPicker), pick which one is
/// active, rename/delete, and preview either staged clip full-screen.
/// Modeled on Danial's Android app's Inputs slot spinner (SitaraPaths.kt /
/// MainActivity.kt refreshInputSpinner/pickInput), adapted to a SwiftUI
/// preset list since PhotosPicker (not a raw SAF file picker) is this
/// app's clip source.
struct ClipPickerView: View {
    @ObservedObject var library: ClipLibrary
    @State private var isCreatingPreset = false
    @State private var newPresetName = ""
    @State private var viewerClip: (title: String, url: URL)?

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                SectionHeader(icon: "film.stack", title: "Test Clips", trailing: AnyView(
                    Button {
                        newPresetName = "Clip \(library.presets.count + 1)"
                        isCreatingPreset = true
                    } label: {
                        Label("New", systemImage: "plus.circle.fill")
                    }
                    .font(.subheadline)
                ))

                if library.presets.isEmpty {
                    emptyState
                } else {
                    VStack(spacing: 8) {
                        ForEach(library.presets) { preset in
                            PresetRow(
                                preset: preset,
                                isActive: library.activePresetID == preset.id,
                                library: library,
                                onSelect: { library.activePresetID = preset.id },
                                onPlay: { title, url in viewerClip = (title, url) }
                            )
                        }
                    }
                }

                if library.activePresetID == nil {
                    Text("Using the bundled sample clip. Select a preset above, or add a new one, to test your own footage.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .sheet(isPresented: $isCreatingPreset) {
            NewPresetSheet(library: library, name: $newPresetName)
        }
        .fullScreenCover(item: Binding(
            get: { viewerClip.map { IdentifiableClip(title: $0.title, url: $0.url) } },
            set: { if $0 == nil { viewerClip = nil } }
        )) { clip in
            VideoViewerSheet(title: clip.title, url: clip.url)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 6) {
            Image(systemName: "film")
                .font(.title2)
                .foregroundStyle(.secondary)
            Text("No clips staged yet")
                .font(.subheadline.weight(.medium))
            Text("Tap New to pick a masked video + mask from Photos")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
    }
}

private struct IdentifiableClip: Identifiable {
    let title: String
    let url: URL
    var id: String { url.path }
}

private struct PresetRow: View {
    let preset: ClipPreset
    let isActive: Bool
    @ObservedObject var library: ClipLibrary
    let onSelect: () -> Void
    let onPlay: (String, URL) -> Void

    @State private var maskedPickerItem: PhotosPickerItem?
    @State private var maskPickerItem: PhotosPickerItem?
    @State private var characterPickerItem: PhotosPickerItem?
    @State private var characterAlphaPickerItem: PhotosPickerItem?
    @State private var isStaging = false
    @State private var stageError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button(action: onSelect) {
                    HStack(spacing: 8) {
                        Image(systemName: isActive ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(isActive ? Theme.accent : .secondary)
                        Text(preset.name)
                            .font(.subheadline.weight(.medium))
                    }
                }
                .buttonStyle(.plain)

                Spacer()

                Badge(text: preset.isComplete ? "Ready" : "Incomplete",
                      color: preset.isComplete ? Theme.success : Theme.warning)

                Menu {
                    Button(role: .destructive) { library.delete(preset) } label: {
                        Label("Delete", systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 8) {
                slotButton(label: "Masked Video", isStaged: FileManager.default.fileExists(atPath: preset.maskedVideoURL.path)) {
                    onPlay("\(preset.name) — Masked", preset.maskedVideoURL)
                }
                PhotosPicker(selection: $maskedPickerItem, matching: .videos) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                }
                .font(.subheadline)

                slotButton(label: "Mask", isStaged: FileManager.default.fileExists(atPath: preset.maskURL.path)) {
                    onPlay("\(preset.name) — Mask", preset.maskURL)
                }
                PhotosPicker(selection: $maskPickerItem, matching: .videos) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                }
                .font(.subheadline)
            }

            // Character + character-alpha slots -- OPTIONAL, unlike the two
            // above: a preset with only masked+mask staged is still fully
            // usable (Final Compositing falls back to
            // Compositor.placeholderCharacter, exactly as before this
            // feature existed). These two rows never affect the
            // "Ready"/"Incomplete" badge above, which is intentionally
            // still keyed off `preset.isComplete` (masked+mask only) --
            // see ClipPreset.isComplete/.hasCharacter in ClipLibrary.swift.
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    slotButton(label: "Character", isStaged: FileManager.default.fileExists(atPath: preset.characterVideoURL.path)) {
                        onPlay("\(preset.name) — Character", preset.characterVideoURL)
                    }
                    PhotosPicker(selection: $characterPickerItem, matching: .videos) {
                        Image(systemName: "arrow.triangle.2.circlepath")
                    }
                    .font(.subheadline)

                    slotButton(label: "Character Alpha", isStaged: FileManager.default.fileExists(atPath: preset.characterAlphaURL.path)) {
                        onPlay("\(preset.name) — Character Alpha", preset.characterAlphaURL)
                    }
                    PhotosPicker(selection: $characterAlphaPickerItem, matching: .videos) {
                        Image(systemName: "arrow.triangle.2.circlepath")
                    }
                    .font(.subheadline)
                }
                Text("(optional — uses placeholder character if empty)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if isStaging {
                ProgressView().controlSize(.small)
            }
            if let stageError {
                Text(stageError).font(.caption2).foregroundStyle(Theme.danger)
            }
        }
        .padding(10)
        .background(isActive ? Theme.accent.opacity(0.08) : Color.clear, in: RoundedRectangle(cornerRadius: Theme.smallCornerRadius))
        .onChange(of: maskedPickerItem) { _, item in stage(item, into: .maskedVideo) { maskedPickerItem = nil } }
        .onChange(of: maskPickerItem) { _, item in stage(item, into: .mask) { maskPickerItem = nil } }
        .onChange(of: characterPickerItem) { _, item in stage(item, into: .character) { characterPickerItem = nil } }
        .onChange(of: characterAlphaPickerItem) { _, item in stage(item, into: .characterAlpha) { characterAlphaPickerItem = nil } }
    }

    private func slotButton(label: String, isStaged: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: isStaged ? "play.circle.fill" : "circle.dashed")
                Text(label)
            }
            .font(.caption)
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(!isStaged)
    }

    /// `clearReset` resets whichever `@State` picker binding triggered this
    /// stage back to nil once staging finishes (success or failure), so a
    /// re-pick of the same asset still fires `.onChange` -- a plain
    /// closure over the specific `@State` var, rather than a KeyPath, since
    /// `@State` property-wrapper storage isn't KeyPath-addressable from a
    /// plain struct method. Shared by all 4 slots instead of 4
    /// near-identical copies (the old 2-slot version special-cased
    /// `asMask` inline; that doesn't scale to 4 slots cleanly).
    private func stage(_ item: PhotosPickerItem?, into slot: ClipSlot, clearReset: @escaping () -> Void) {
        guard let item else { return }
        isStaging = true
        stageError = nil
        Task {
            do {
                try await library.stage(item, into: slot, in: preset)
            } catch {
                stageError = error.localizedDescription
            }
            isStaging = false
            clearReset()
        }
    }
}

private struct NewPresetSheet: View {
    @ObservedObject var library: ClipLibrary
    @Binding var name: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Preset name") {
                    TextField("e.g. near-static", text: $name)
                }
                Section {
                    Text("After creating, use the ⟲ buttons on the clip row to stage a Masked Video and a Mask from Photos.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("New Clip Preset")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create") {
                        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
                        let preset = library.createPreset(named: trimmed.isEmpty ? "Clip" : trimmed)
                        library.activePresetID = preset.id
                        dismiss()
                    }
                }
            }
        }
        .presentationDetents([.medium])
    }
}
