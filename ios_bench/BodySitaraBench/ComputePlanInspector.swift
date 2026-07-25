import CoreML
import Foundation

/// Reports what compute-unit configuration was REQUESTED for a model, and
/// what's actually AVAILABLE on this device -- a weaker signal than true
/// per-operation ANE/GPU/CPU attribution, but one that compiles against a
/// stable, well-documented API.
///
/// BACKGROUND: this originally attempted true per-operation attribution via
/// MLComputePlan (op-by-op device usage), matching the verification
/// discipline the Android RIFE investigation needed -- that investigation
/// found every attempted QNN/HTP "NPU" path was silently running 100% on
/// CPU, caught only via verbose adb logcat after the fact, so reporting an
/// "ANE" latency number here with no equivalent check would risk the same
/// mistake. However, MLComputePlan's exact Swift method/case names could
/// not be pinned down: three real CI compile failures in a row (2026-07-26)
/// each guessed a different name ('.mlProgram'/'computeDevice' ->
/// '.program'/'computeDeviceUsage' -> '.program'/
/// 'computeDeviceUsageForMLProgramOperation'), none of which matched the
/// real API, and Apple's own documentation pages returned no fetchable
/// content to verify against directly. Rather than keep guessing
/// undocumented API surface across further CI cycles, this component was
/// simplified to what CAN be verified from documented, stable APIs:
///   - MLModelConfiguration.computeUnits: what was REQUESTED
///   - MLModel.availableComputeDevices: what's available on this hardware
///     (a real, older, well-documented property)
/// Neither of these confirms actual per-op ANE execution the way
/// MLComputePlan would -- that stronger check needs Instruments' Core ML
/// template on a real Mac (out of scope while working Mac-free via CI).
/// This limitation should be stated explicitly when reporting benchmark
/// numbers: "ANE requested via computeUnits=.all; per-operation placement
/// not independently verified in this pass."
enum ComputePlanInspector {
    struct RequestSummary {
        let requestedComputeUnits: String
        let availableDeviceDescriptions: [String]

        var summary: String {
            "requested=\(requestedComputeUnits) available=[\(availableDeviceDescriptions.joined(separator: ", "))]"
        }
    }

    static func inspect(model: MLModel, configuration: MLModelConfiguration) -> RequestSummary {
        let requested: String
        switch configuration.computeUnits {
        case .all: requested = "all (CPU+GPU+ANE)"
        case .cpuOnly: requested = "cpuOnly"
        case .cpuAndGPU: requested = "cpuAndGPU"
        case .cpuAndNeuralEngine: requested = "cpuAndNeuralEngine"
        @unknown default: requested = "unknown"
        }

        let devices = MLModel.availableComputeDevices.map { device -> String in
            String(describing: device)
        }

        return RequestSummary(requestedComputeUnits: requested, availableDeviceDescriptions: devices)
    }
}
