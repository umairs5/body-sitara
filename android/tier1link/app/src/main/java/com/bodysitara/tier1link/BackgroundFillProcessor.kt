package com.bodysitara.tier1link

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.util.Log
import java.io.File
import java.nio.ByteBuffer

private const val TAG = "BackgroundFillProcessor"

/**
 * Drives LamaDilatedFiller across every frame of a clip's output_rtm.mp4 +
 * mask.mp4 (Tier 1's dense-export bundle), producing a filled-background
 * video. This is the real Tier 2B-1 implementation: on-device, per-frame,
 * no temporal-median pre-pass (per the plan's corrected design).
 *
 * Frame extraction uses MediaMetadataRetriever (simple, seek-based -- not
 * the fastest approach for a full clip, but robust and doesn't require
 * managing MediaCodec decoder state directly). Encoding uses MediaCodec +
 * MediaMuxer directly, since there's no simpler stable Android API for
 * "write these Bitmaps as an H.264 mp4."
 */
class BackgroundFillProcessor(private val filler: LamaDilatedFiller) {

    /**
     * progressCallback: (framesDone, totalFrames) after each frame.
     * Returns the output video file.
     */
    suspend fun processVideo(
        videoPath: String,
        maskPath: String,
        outputPath: String,
        growMaskPx: Int = 10,
        progressCallback: ((Int, Int) -> Unit)? = null,
    ): File {
        val videoRetriever = MediaMetadataRetriever()
        val maskRetriever = MediaMetadataRetriever()
        videoRetriever.setDataSource(videoPath)
        maskRetriever.setDataSource(maskPath)

        val durationMs = videoRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            ?.toLongOrNull() ?: 0L
        val frameRate = videoRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
            ?.toFloatOrNull() ?: 30f
        val width = videoRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
            ?.toIntOrNull() ?: 0
        val height = videoRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
            ?.toIntOrNull() ?: 0

        val totalFrames = ((durationMs / 1000.0) * frameRate).toInt().coerceAtLeast(1)
        val frameIntervalUs = (1_000_000.0 / frameRate).toLong()

        Log.d(TAG, "Video: ${width}x${height}, ~$totalFrames frames @ ${frameRate}fps")

        val encoder = FrameVideoEncoder(outputPath, width, height, frameRate)

        var frameIdx = 0
        var timeUs = 0L
        try {
            while (timeUs < durationMs * 1000) {
                val frameBitmap = videoRetriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                val maskBitmap = maskRetriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)

                if (frameBitmap != null && maskBitmap != null) {
                    val filled = filler.fillFrame(frameBitmap, maskBitmap, growMaskPx)
                    encoder.encodeFrame(filled)
                }

                frameIdx++
                progressCallback?.invoke(frameIdx, totalFrames)
                timeUs += frameIntervalUs
            }
        } finally {
            videoRetriever.release()
            maskRetriever.release()
            encoder.finish()
        }

        return File(outputPath)
    }
}

/**
 * Minimal H.264/mp4 encoder for a sequence of Bitmaps, via MediaCodec +
 * MediaMuxer. Android has no higher-level stable API for this that isn't
 * either deprecated or a much larger dependency (e.g. a full FFmpeg
 * build) -- this is the standard, if verbose, direct approach.
 */
private class FrameVideoEncoder(outputPath: String, private val width: Int, private val height: Int, frameRate: Float) {
    private val codec: MediaCodec
    private val muxer: MediaMuxer
    private var trackIndex = -1
    private var muxerStarted = false
    private var frameIndex = 0L
    private val frameDurationUs = (1_000_000.0 / frameRate).toLong()

    init {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            // NOTE: COLOR_FormatYUV420Flexible is a hint, not a guarantee of
            // a specific byte layout (NV21 vs NV12 vs I420 all qualify) --
            // different devices/encoders can expect different exact planar
            // arrangements despite reporting the same "flexible" format.
            // bitmapToNv21() below produces one specific fixed layout
            // (semi-planar NV21: Y plane, then interleaved V,U). This is a
            // REAL, UNVERIFIED RISK for device compatibility -- confirmed
            // working only insofar as it needs a real on-device test; if
            // colors come out wrong (commonly a green/purple tint) on a
            // given device, this mismatch is the first thing to check.
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, width * height * 4)
            setInteger(MediaFormat.KEY_FRAME_RATE, frameRate.toInt().coerceAtLeast(1))
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
        muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    }

    fun encodeFrame(bitmap: Bitmap) {
        val inputBufferId = codec.dequeueInputBuffer(10_000)
        if (inputBufferId >= 0) {
            val inputBuffer = codec.getInputBuffer(inputBufferId)
            inputBuffer?.clear()
            inputBuffer?.put(bitmapToNv21(bitmap))
            val ptsUs = frameIndex * frameDurationUs
            codec.queueInputBuffer(inputBufferId, 0, inputBuffer?.position() ?: 0, ptsUs, 0)
            frameIndex++
        }
        drainEncoder(false)
    }

    fun finish() {
        val inputBufferId = codec.dequeueInputBuffer(10_000)
        if (inputBufferId >= 0) {
            codec.queueInputBuffer(inputBufferId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        }
        drainEncoder(true)
        codec.stop()
        codec.release()
        if (muxerStarted) muxer.stop()
        muxer.release()
    }

    private fun drainEncoder(endOfStream: Boolean) {
        val bufferInfo = MediaCodec.BufferInfo()
        while (true) {
            val outputBufferId = codec.dequeueOutputBuffer(bufferInfo, 10_000)
            when {
                outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) return
                }
                outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    trackIndex = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                    muxerStarted = true
                }
                outputBufferId >= 0 -> {
                    val outputBuffer = codec.getOutputBuffer(outputBufferId)
                    if (outputBuffer != null && bufferInfo.size > 0 && muxerStarted) {
                        outputBuffer.position(bufferInfo.offset)
                        outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                        muxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                    }
                    codec.releaseOutputBuffer(outputBufferId, false)
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }
    }

    private fun bitmapToNv21(bitmap: Bitmap): ByteArray {
        val scaled = if (bitmap.width != width || bitmap.height != height)
            Bitmap.createScaledBitmap(bitmap, width, height, true) else bitmap
        val argb = IntArray(width * height)
        scaled.getPixels(argb, 0, width, 0, 0, width, height)

        val yuv = ByteArray(width * height * 3 / 2)
        var yIndex = 0
        var uvIndex = width * height

        for (j in 0 until height) {
            for (i in 0 until width) {
                val pixel = argb[j * width + i]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF

                val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                yuv[yIndex++] = y.coerceIn(0, 255).toByte()

                if (j % 2 == 0 && i % 2 == 0) {
                    val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                    val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                    yuv[uvIndex++] = v.coerceIn(0, 255).toByte()
                    yuv[uvIndex++] = u.coerceIn(0, 255).toByte()
                }
            }
        }
        return yuv
    }
}
