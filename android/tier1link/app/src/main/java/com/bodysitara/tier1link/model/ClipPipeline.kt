package com.bodysitara.tier1link.model

/**
 * One clip's state as it moves through the full Tier1->Tier2 pipeline:
 *
 *   Pull (from Pi)  ->  [BackgroundFill (on-phone) || CloudGenerate (RunPod)]  ->  Composite  ->  Done
 *
 * BackgroundFill and CloudGenerate run concurrently (see the plan doc's
 * correction: the phone doesn't wait for one before starting the other),
 * so they're modeled as siblings under one PROCESSING stage rather than
 * two separate sequential steps.
 *
 * As of this UI build: only PULL is backed by real network code
 * (Tier1LinkClient). BACKGROUND_FILL and CLOUD_GENERATE have no real
 * implementation yet (no LaMa integration, no RunPod gateway) -- their
 * UI states exist and are wired to show real progress once that code
 * exists, but today they'll simply never leave PENDING/IN_PROGRESS
 * unless a caller drives them manually (e.g. a future debug/simulate
 * hook). COMPOSITE is entirely unimplemented (no on-device compositor
 * exists yet either).
 */
enum class StageStatus {
    PENDING,
    IN_PROGRESS,
    DONE,
    FAILED,
    NOT_CONNECTED,  // stage exists in the pipeline but its backend isn't built yet
}

enum class ClipStage {
    PULL,
    PROCESSING,  // BackgroundFill + CloudGenerate, running concurrently
    COMPOSITE,
    DONE,
}

data class StageState(
    val status: StageStatus = StageStatus.PENDING,
    val progressPercent: Int? = null,  // null = indeterminate / not applicable
    val message: String? = null,
)

data class ClipPipelineState(
    val clipId: String,
    val totalFrames: Int,
    val numSlots: Int,

    val pull: StageState = StageState(),
    val backgroundFill: StageState = StageState(status = StageStatus.NOT_CONNECTED),
    val cloudGenerate: StageState = StageState(status = StageStatus.NOT_CONNECTED),
    val composite: StageState = StageState(status = StageStatus.NOT_CONNECTED),

    /** Local file path to the pulled output_rtm.mp4, once PULL completes. */
    val incomingVideoPath: String? = null,

    /** Local file path to the local mask.mp4, once PULL completes -- needed
     * as backgroundFill's second input. */
    val maskVideoPath: String? = null,

    /** Local file path to the background-filled video, once backgroundFill
     * completes. Real (LaMa-Dilated on-device inference), not a placeholder. */
    val filledVideoPath: String? = null,

    /** Local file path to the final composited video, once COMPOSITE completes. Always
     * null today -- nothing produces this yet. */
    val finalVideoPath: String? = null,
) {
    val currentStage: ClipStage
        get() = when {
            composite.status == StageStatus.DONE -> ClipStage.DONE
            pull.status == StageStatus.DONE -> ClipStage.PROCESSING
            else -> ClipStage.PULL
        }

    /** Whichever single stage is most relevant to show as a compact icon (e.g. a
     * clip-list row) -- the stage currently active, or the last one completed. */
    val leadStageState: StageState
        get() = when (currentStage) {
            ClipStage.PULL -> pull
            ClipStage.PROCESSING -> if (backgroundFill.status == StageStatus.DONE && cloudGenerate.status == StageStatus.DONE)
                StageState(status = StageStatus.DONE) else StageState(status = StageStatus.IN_PROGRESS)
            ClipStage.COMPOSITE -> composite
            ClipStage.DONE -> composite
        }

    /** One-line status for the clip-list row. */
    val summary: String
        get() = when (currentStage) {
            ClipStage.PULL -> when (pull.status) {
                StageStatus.IN_PROGRESS -> "Downloading${pull.progressPercent?.let { " $it%" } ?: "..."}"
                StageStatus.FAILED -> "Download failed"
                else -> "Queued"
            }
            ClipStage.PROCESSING -> {
                val fill = backgroundFill.status
                val gen = cloudGenerate.status
                when {
                    fill == StageStatus.NOT_CONNECTED && gen == StageStatus.NOT_CONNECTED ->
                        "Ready to process (not yet connected)"
                    fill == StageStatus.DONE && gen == StageStatus.DONE -> "Ready to composite"
                    else -> "Filling background & generating avatar..."
                }
            }
            ClipStage.COMPOSITE -> "Compositing..."
            ClipStage.DONE -> "Ready"
        }
}
