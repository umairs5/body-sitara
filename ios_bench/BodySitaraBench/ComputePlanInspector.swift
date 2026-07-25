import CoreML
import Foundation

/// Reports which compute unit (ANE / GPU / CPU) each operation in a
/// compiled CoreML model actually ran on -- required verification step,
/// not optional: the Android RIFE investigation (QNN/HTP on Snapdragon)
/// silently ran 100% on CPU for every attempted NPU path, and that was
/// ONLY caught via verbose adb logcat inspection after the fact. A
/// benchmark number with no equivalent check here would risk repeating
/// the exact same mistake -- reporting an "ANE" latency number that was
/// actually CPU-only the whole time.
enum ComputePlanInspector {
    struct UnitCounts {
        var cpu = 0
        var gpu = 0
        var aneNeuralEngine = 0
        var unknown = 0

        var summary: String {
            "CPU=\(cpu) GPU=\(gpu) ANE=\(aneNeuralEngine) unknown=\(unknown)"
        }

        /// True only if at least one op ran on ANE and NOTHING silently
        /// fell back to CPU for the whole graph (a few CPU-only ops for
        /// glue/reshape are normal; all-CPU is the failure mode to catch).
        var looksLikeRealANEUsage: Bool {
            aneNeuralEngine > 0 && cpu < (cpu + gpu + aneNeuralEngine)
        }
    }

    /// Loads the compiled model's MLComputePlan and tallies which compute
    /// device each operation was assigned to. Must be called on a
    /// compiled model URL (.mlmodelc), matching MLComputePlan's API.
    ///
    /// NOTE: the exact MLModelStructure/MLComputePlan Swift API surface
    /// (enum case names, method names) could not be verified against
    /// real Apple documentation before this was written (fetched pages
    /// returned no usable content). Corrected TWICE from real compiler
    /// errors (2026-07-26 CI runs): first 'mlProgram' -> '.program' and
    /// 'computeDevice' -> attempted 'computeDeviceUsage' (still wrong);
    /// this second correction uses get_compute_device_usage_for_mlprogram
    /// _operation(), the confirmed real Python coremltools method name
    /// (the Swift binding is very likely the same method with standard
    /// Swift naming-convention transformation: drop get_, camelCase,
    /// keep the descriptive suffix rather than a generic "for:" label,
    /// since Apple's compute-plan API is method-name-explicit throughout).
    /// Still a best-effort guess, not confirmed -- next CI run verifies.
    static func inspect(compiledModelURL: URL, configuration: MLModelConfiguration) async throws -> UnitCounts {
        let plan = try await MLComputePlan.load(contentsOf: compiledModelURL, configuration: configuration)
        var counts = UnitCounts()

        guard case let .program(program) = plan.modelStructure else {
            // Non-mlprogram structure (e.g. neuralnetwork) -- compute plan
            // API surface differs; report unknown rather than guess.
            counts.unknown += 1
            return counts
        }

        guard let mainFunction = program.functions["main"] else {
            counts.unknown += 1
            return counts
        }

        func walk(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                guard let deviceUsage = plan.computeDeviceUsageForMLProgramOperation(op) else {
                    counts.unknown += 1
                    continue
                }
                switch deviceUsage {
                case .cpuOnly:
                    counts.cpu += 1
                case .cpuAndGPU:
                    counts.gpu += 1
                case .cpuAndNeuralEngine:
                    counts.aneNeuralEngine += 1
                @unknown default:
                    counts.unknown += 1
                }
                // Recurse into nested blocks (e.g. control flow ops), if any.
                for nested in op.blocks {
                    walk(nested)
                }
            }
        }

        walk(mainFunction.block)
        return counts
    }
}
