import CoreTransferable
import Foundation
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// A named, staged pair of test clips (masked video + mask) the benchmark
/// can run against -- lets you build up a small library of real dataset
/// clips (e.g. "near-static", "bystander-movement") and switch between
/// them, rather than editing Swift/rebuilding to test a different clip.
/// Modeled on Danial's Android app's "Inputs" slot system (SitaraPaths.kt/
/// MainActivity.kt: pick a video, stage it into a canonical named slot),
/// adapted to SwiftUI's PhotosPicker + a small on-disk preset library
/// instead of his SAF file-slot spinner.
struct ClipPreset: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    let createdAt: Date

    var maskedVideoURL: URL { clipPresetDir(id).appendingPathComponent("masked_video.mp4") }
    var maskURL: URL { clipPresetDir(id).appendingPathComponent("mask.mp4") }

    /// OPTIONAL slots: a real cloud-generated (WanAnimate) synthetic
    /// character video + its matching alpha matte, for testing Final
    /// Compositing against real character content instead of the
    /// placeholder ellipse (Compositor.placeholderCharacter). A preset
    /// with only masked+mask staged is still `isComplete` -- these two
    /// slots are additive and never gate that flag (see `isComplete`
    /// below, deliberately unchanged) -- BenchmarkView falls back to the
    /// placeholder whenever `hasCharacter` is false, exactly as it always
    /// has.
    var characterVideoURL: URL { clipPresetDir(id).appendingPathComponent("character.mp4") }
    var characterAlphaURL: URL { clipPresetDir(id).appendingPathComponent("character_alpha.mp4") }

    var isComplete: Bool {
        FileManager.default.fileExists(atPath: maskedVideoURL.path) &&
        FileManager.default.fileExists(atPath: maskURL.path)
    }

    /// True only when BOTH the character video and its alpha matte are
    /// staged -- one without the other can't be composited (a character
    /// with no alpha has no valid cutout region; an alpha with no
    /// character has nothing to cut out), so BenchmarkView treats a
    /// partially-staged pair the same as "no character staged" and falls
    /// back to the placeholder rather than guessing.
    var hasCharacter: Bool {
        FileManager.default.fileExists(atPath: characterVideoURL.path) &&
        FileManager.default.fileExists(atPath: characterAlphaURL.path)
    }
}

/// Which on-disk slot a picked PhotosPickerItem should be staged into.
/// Replaces the old `asMask: Bool` (which only distinguished 2 slots) now
/// that a preset has 4 possible slots. `asMask`-style call sites are
/// migrated to pass `.mask`/`.maskedVideo` explicitly.
enum ClipSlot {
    case maskedVideo
    case mask
    case character
    case characterAlpha

    func url(in preset: ClipPreset) -> URL {
        switch self {
        case .maskedVideo: return preset.maskedVideoURL
        case .mask: return preset.maskURL
        case .character: return preset.characterVideoURL
        case .characterAlpha: return preset.characterAlphaURL
        }
    }
}

/// Free function (NOT a ClipLibrary static method): ClipPreset's computed
/// URL properties are nonisolated (plain struct, no actor), so this must
/// stay nonisolated too, or every URL access from a nonisolated context
/// (e.g. FileManager checks off the main actor) fails to compile with
/// "call to main actor-isolated static method in a synchronous
/// nonisolated context" -- hit for real in CI (2026-07-31) when this used
/// to be a `@MainActor`-inherited static method on ClipLibrary. Pure path
/// math, no actor-isolated state involved, so nonisolated is correct.
func clipPresetDir(_ id: UUID) -> URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("ClipPresets", isDirectory: true)
        .appendingPathComponent(id.uuidString, isDirectory: true)
}

@MainActor
final class ClipLibrary: ObservableObject {
    @Published private(set) var presets: [ClipPreset] = []
    @Published var activePresetID: UUID? {
        didSet { UserDefaults.standard.set(activePresetID?.uuidString, forKey: Self.activeKey) }
    }

    private static let indexFile = "clip_presets.json"
    private static let activeKey = "activeClipPresetID"

    private var documentsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private var presetsRootDir: URL { documentsDir.appendingPathComponent("ClipPresets", isDirectory: true) }
    private var indexURL: URL { presetsRootDir.appendingPathComponent(Self.indexFile) }

    init() {
        load()
        if let saved = UserDefaults.standard.string(forKey: Self.activeKey), let uuid = UUID(uuidString: saved) {
            activePresetID = uuid
        }
    }

    /// The masked/mask URLs the benchmark should read from: the active
    /// preset if one is selected and fully staged, otherwise nil (caller
    /// falls back to the bundled sample clip, matching the app's original
    /// behavior).
    var activeClip: (masked: URL, mask: URL)? {
        guard let id = activePresetID, let preset = presets.first(where: { $0.id == id }), preset.isComplete else { return nil }
        return (preset.maskedVideoURL, preset.maskURL)
    }

    /// The real synthetic-character video + alpha matte URLs, if the
    /// active preset has both staged. nil whenever no preset is active,
    /// the active preset isn't `isComplete` (matches `activeClip`'s own
    /// gate -- no point staging a character over a clip that has no
    /// masked/mask pair to reconstruct a background from), or the
    /// character/alpha pair isn't fully staged -- callers (BenchmarkView)
    /// treat nil as "use the placeholder", unchanged from today's
    /// behavior.
    var activeCharacter: (character: URL, alpha: URL)? {
        guard let id = activePresetID, let preset = presets.first(where: { $0.id == id }), preset.isComplete, preset.hasCharacter else { return nil }
        return (preset.characterVideoURL, preset.characterAlphaURL)
    }

    var activePreset: ClipPreset? {
        presets.first(where: { $0.id == activePresetID })
    }

    func createPreset(named name: String) -> ClipPreset {
        let preset = ClipPreset(id: UUID(), name: name, createdAt: Date())
        try? FileManager.default.createDirectory(at: clipPresetDir(preset.id), withIntermediateDirectories: true)
        presets.append(preset)
        save()
        return preset
    }

    func delete(_ preset: ClipPreset) {
        try? FileManager.default.removeItem(at: clipPresetDir(preset.id))
        presets.removeAll { $0.id == preset.id }
        if activePresetID == preset.id { activePresetID = nil }
        save()
    }

    func rename(_ preset: ClipPreset, to newName: String) {
        guard let idx = presets.firstIndex(where: { $0.id == preset.id }) else { return }
        presets[idx].name = newName
        save()
    }

    /// Copies a picked PhotosPicker video into one of the preset's 4
    /// canonical slots (masked video, mask, character, character alpha).
    /// Overwrites any previously staged file in that slot -- matches
    /// Danial's "re-pick overwrites a slot" behavior.
    ///
    /// Loads via VideoPickerFile (a file-URL Transferable), NOT
    /// `loadTransferable(type: Data.self)`: PhotosPicker video items don't
    /// reliably vend a Data transfer representation, and reading a
    /// multi-hundred-MB clip fully into memory as Data before writing it
    /// back out would be wasteful even if they did. The file-representation
    /// transfer streams straight to a temp file that we then move/copy.
    func stage(_ item: PhotosPickerItem, into slot: ClipSlot, in preset: ClipPreset) async throws {
        guard let received = try await item.loadTransferable(type: VideoPickerFile.self) else {
            throw NSError(domain: "ClipLibrary", code: 1, userInfo: [NSLocalizedDescriptionKey: "could not load video from picker"])
        }
        let dest = slot.url(in: preset)
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.copyItem(at: received.url, to: dest)
        try? FileManager.default.removeItem(at: received.url)   // received.url is a transient temp copy owned by us
        objectWillChange.send()   // isComplete/hasCharacter on the struct are derived from disk state, not stored -- force a UI refresh
    }

    private func load() {
        guard let data = try? Data(contentsOf: indexURL),
              let decoded = try? JSONDecoder().decode([ClipPreset].self, from: data) else { return }
        presets = decoded
    }

    private func save() {
        try? FileManager.default.createDirectory(at: presetsRootDir, withIntermediateDirectories: true)
        guard let data = try? JSONEncoder().encode(presets) else { return }
        try? data.write(to: indexURL)
    }
}

/// Transferable wrapper that receives a PhotosPicker video as a temp file
/// URL (via the `.file` import representation) rather than in-memory
/// Data -- the correct pattern for large video assets. The system copies
/// the asset into a caller-owned temp location for us; ClipLibrary.stage
/// then moves that temp copy into the app's own ClipPresets directory.
struct VideoPickerFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { file in
            SentTransferredFile(file.url)
        } importing: { received in
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension)
            if FileManager.default.fileExists(atPath: tempURL.path) {
                try FileManager.default.removeItem(at: tempURL)
            }
            try FileManager.default.copyItem(at: received.file, to: tempURL)
            return Self(url: tempURL)
        }
    }
}
