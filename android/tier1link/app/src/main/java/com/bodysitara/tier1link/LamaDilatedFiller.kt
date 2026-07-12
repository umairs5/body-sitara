package com.bodysitara.tier1link

import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.TimeUnit

private const val TAG = "LamaDilatedFiller"

/** Where the model files live once downloaded (see companion ensureModel()). */
private const val MODEL_ONNX_URL =
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/lama_dilated/releases/v0.57.3/lama_dilated-onnx-float.zip"
private const val REQUIRED_SIZE = 512

/**
 * LaMa-Dilated background inpainting (Tier 2B-1), real on-device inference
 * via ONNX Runtime Mobile. Kotlin port of scripts/lama_dilated_test.py's
 * validated preprocessing (resize to 512x512, mask binarize+dilate,
 * normalize [0,1]) and postprocessing (resize result back to source
 * resolution via bilinear -- Bitmap has no built-in Lanczos, unlike our
 * Python reference; a real, known quality gap vs. the desktop reference
 * implementation, not an oversight).
 *
 * This is the mobile-exportable LaMa variant (dilated convolutions, not
 * FFC/FFT) -- big-lama (the higher-quality FFC model validated extensively
 * in the Python reference implementation) is confirmed NOT exportable to
 * ONNX or ExecuTorch; see the project's lama_onnx_export.py /
 * lama_executorch_export.py findings. LaMa-Dilated trades quality for
 * being real and running today.
 */
class LamaDilatedFiller(private val modelPath: String) {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelPath, OrtSession.SessionOptions())

    /**
     * frame: the anonymized (grey-filled) frame.
     * mask: same size as frame, white (255) where the person is.
     * growMaskPx: dilate the mask by this many pixels before inference,
     *     matching the validated Python reference's contamination-boundary
     *     fix (see background_fill.py's LamaBackgroundFiller docstring).
     */
    fun fillFrame(frame: Bitmap, mask: Bitmap, growMaskPx: Int = 10): Bitmap {
        val w = frame.width
        val h = frame.height

        val maskGrown = if (growMaskPx > 0) dilateMask(mask, growMaskPx) else mask

        val frameResized = Bitmap.createScaledBitmap(frame, REQUIRED_SIZE, REQUIRED_SIZE, true)
        val maskResized = Bitmap.createScaledBitmap(maskGrown, REQUIRED_SIZE, REQUIRED_SIZE, false)

        val imageTensor = bitmapToCHWTensor(frameResized)
        val maskTensor = maskToTensor(maskResized)

        val inputs = mapOf("image" to imageTensor, "mask" to maskTensor)
        val results = session.run(inputs)
        val outputTensor = results[0].value as Array<Array<Array<FloatArray>>>
        imageTensor.close()
        maskTensor.close()
        results.close()

        val resultBitmap = chwArrayToBitmap(outputTensor[0])
        val resultFullRes = Bitmap.createScaledBitmap(resultBitmap, w, h, true)

        return blendByMask(frame, resultFullRes, maskGrown)
    }

    fun close() {
        session.close()
    }

    private fun dilateMask(mask: Bitmap, px: Int): Bitmap {
        // Separable box-dilation (horizontal max-pass, then vertical) --
        // O(w*h*px) instead of a naive full 2D window's O(w*h*px^2).
        // Produces a square-kernel dilation, not OpenCV's elliptical kernel
        // (the Python reference) -- a real, documented simplification made
        // to avoid an OpenCV-for-Android dependency for just this one step;
        // close enough at the pixel radii we use (~10px) for this purpose
        // (excluding a thin contaminated mask boundary), not pixel-identical
        // to the desktop reference.
        val w = mask.width
        val h = mask.height
        val src = IntArray(w * h)
        mask.getPixels(src, 0, w, 0, 0, w, h)
        val srcBinary = BooleanArray(w * h) { (src[it] and 0xFF) > 127 }

        val horizontal = BooleanArray(w * h)
        for (y in 0 until h) {
            val rowBase = y * w
            for (x in 0 until w) {
                var found = false
                val xStart = maxOf(0, x - px)
                val xEnd = minOf(w - 1, x + px)
                var xx = xStart
                while (xx <= xEnd && !found) {
                    if (srcBinary[rowBase + xx]) found = true
                    xx++
                }
                horizontal[rowBase + x] = found
            }
        }

        val out = BooleanArray(w * h)
        for (x in 0 until w) {
            for (y in 0 until h) {
                var found = false
                val yStart = maxOf(0, y - px)
                val yEnd = minOf(h - 1, y + px)
                var yy = yStart
                while (yy <= yEnd && !found) {
                    if (horizontal[yy * w + x]) found = true
                    yy++
                }
                out[y * w + x] = found
            }
        }

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val outPixels = IntArray(w * h) { if (out[it]) 0xFFFFFFFF.toInt() else 0xFF000000.toInt() }
        result.setPixels(outPixels, 0, w, 0, 0, w, h)
        return result
    }

    private fun bitmapToCHWTensor(bmp: Bitmap): OnnxTensor {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val buffer = FloatBuffer.allocate(3 * h * w)
        // CHW, channel order R,G,B, normalized [0,1]
        for (c in 0 until 3) {
            for (i in 0 until h * w) {
                val p = pixels[i]
                val v = when (c) {
                    0 -> (p shr 16) and 0xFF
                    1 -> (p shr 8) and 0xFF
                    else -> p and 0xFF
                }
                buffer.put(v / 255f)
            }
        }
        buffer.rewind()
        return OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, longArrayOf(1, 3, h.toLong(), w.toLong()))
    }

    private fun maskToTensor(bmp: Bitmap): OnnxTensor {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val buffer = FloatBuffer.allocate(h * w)
        for (i in 0 until h * w) {
            val v = pixels[i] and 0xFF
            buffer.put(if (v > 127) 1f else 0f)
        }
        buffer.rewind()
        return OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, longArrayOf(1, 1, h.toLong(), w.toLong()))
    }

    private fun chwArrayToBitmap(chw: Array<Array<FloatArray>>): Bitmap {
        val h = chw[0].size
        val w = chw[0][0].size
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val r = (chw[0][y][x].coerceIn(0f, 1f) * 255).toInt()
                val g = (chw[1][y][x].coerceIn(0f, 1f) * 255).toInt()
                val b = (chw[2][y][x].coerceIn(0f, 1f) * 255).toInt()
                pixels[y * w + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        bitmap.setPixels(pixels, 0, w, 0, 0, w, h)
        return bitmap
    }

    private fun blendByMask(original: Bitmap, filled: Bitmap, mask: Bitmap): Bitmap {
        val w = original.width
        val h = original.height
        val origPixels = IntArray(w * h)
        val filledPixels = IntArray(w * h)
        val maskPixels = IntArray(w * h)
        original.getPixels(origPixels, 0, w, 0, 0, w, h)
        filled.getPixels(filledPixels, 0, w, 0, 0, w, h)
        mask.getPixels(maskPixels, 0, w, 0, 0, w, h)

        val out = IntArray(w * h)
        for (i in 0 until w * h) {
            val maskOn = (maskPixels[i] and 0xFF) > 127
            out[i] = if (maskOn) filledPixels[i] else origPixels[i]
        }
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    companion object {
        /**
         * Downloads and extracts the LaMa-Dilated ONNX model into [destDir]
         * if not already present. Returns the path to the .onnx graph file
         * (the .data external-weights file must sit alongside it -- ONNX
         * Runtime resolves that reference by relative path automatically).
         */
        suspend fun ensureModel(destDir: File, onProgress: ((Long, Long) -> Unit)? = null): String {
            val onnxFile = File(destDir, "lama_dilated.onnx")
            val dataFile = File(destDir, "lama_dilated.data")
            if (onnxFile.exists() && dataFile.exists()) {
                return onnxFile.absolutePath
            }

            destDir.mkdirs()
            val zipFile = File(destDir, "lama_dilated_onnx.zip")
            val http = OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(10, TimeUnit.MINUTES)
                .build()
            val req = Request.Builder().url(MODEL_ONNX_URL).build()
            http.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) throw java.io.IOException("Model download failed: ${resp.code}")
                val body = resp.body ?: throw java.io.IOException("Empty model download response")
                val total = body.contentLength()
                var read = 0L
                zipFile.outputStream().use { out ->
                    body.byteStream().use { input ->
                        val buffer = ByteArray(64 * 1024)
                        while (true) {
                            val n = input.read(buffer)
                            if (n == -1) break
                            out.write(buffer, 0, n)
                            read += n
                            onProgress?.invoke(read, total)
                        }
                    }
                }
            }

            Log.d(TAG, "Extracting model zip...")
            java.util.zip.ZipInputStream(zipFile.inputStream()).use { zis ->
                var entry = zis.nextEntry
                while (entry != null) {
                    if (!entry.isDirectory) {
                        val outFile = File(destDir, File(entry.name).name)
                        outFile.outputStream().use { out -> zis.copyTo(out) }
                    }
                    entry = zis.nextEntry
                }
            }
            zipFile.delete()

            if (!onnxFile.exists() || !dataFile.exists()) {
                throw java.io.IOException("Model extraction did not produce expected files")
            }
            return onnxFile.absolutePath
        }
    }
}
