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

    var maskedVideoURL: URL { ClipLibrary.presetDir(id).appendingPathComponent("masked_video.mp4") }
    var maskURL: URL { ClipLibrary.presetDir(id).appendingPathComponent("mask.mp4") }

    var isComplete: Bool {
        FileManager.default.fileExists(atPath: maskedVideoURL.path) &&
        FileManager.default.fileExists(atPath: maskURL.path)
    }
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

    /// Static so ClipPreset's computed URLs don't need an instance.
    fileprivate static func presetDir(_ id: UUID) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ClipPresets", isDirectory: true)
            .appendingPathComponent(id.uuidString, isDirectory: true)
    }

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

    var activePreset: ClipPreset? {
        presets.first(where: { $0.id == activePresetID })
    }

    func createPreset(named name: String) -> ClipPreset {
        let preset = ClipPreset(id: UUID(), name: name, createdAt: Date())
        try? FileManager.default.createDirectory(at: Self.presetDir(preset.id), withIntermediateDirectories: true)
        presets.append(preset)
        save()
        return preset
    }

    func delete(_ preset: ClipPreset) {
        try? FileManager.default.removeItem(at: Self.presetDir(preset.id))
        presets.removeAll { $0.id == preset.id }
        if activePresetID == preset.id { activePresetID = nil }
        save()
    }

    func rename(_ preset: ClipPreset, to newName: String) {
        guard let idx = presets.firstIndex(where: { $0.id == preset.id }) else { return }
        presets[idx].name = newName
        save()
    }

    /// Copies a picked PhotosPicker video into the preset's canonical
    /// masked-video or mask slot. Overwrites any previously staged file in
    /// that slot -- matches Danial's "re-pick overwrites a slot" behavior.
    ///
    /// Loads via VideoPickerFile (a file-URL Transferable), NOT
    /// `loadTransferable(type: Data.self)`: PhotosPicker video items don't
    /// reliably vend a Data transfer representation, and reading a
    /// multi-hundred-MB clip fully into memory as Data before writing it
    /// back out would be wasteful even if they did. The file-representation
    /// transfer streams straight to a temp file that we then move/copy.
    func stage(_ item: PhotosPickerItem, asMask: Bool, in preset: ClipPreset) async throws {
        guard let received = try await item.loadTransferable(type: VideoPickerFile.self) else {
            throw NSError(domain: "ClipLibrary", code: 1, userInfo: [NSLocalizedDescriptionKey: "could not load video from picker"])
        }
        let dest = asMask ? preset.maskURL : preset.maskedVideoURL
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.copyItem(at: received.url, to: dest)
        try? FileManager.default.removeItem(at: received.url)   // received.url is a transient temp copy owned by us
        objectWillChange.send()   // isComplete on the struct is derived from disk state, not stored -- force a UI refresh
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
