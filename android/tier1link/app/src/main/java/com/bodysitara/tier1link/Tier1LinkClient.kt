package com.bodysitara.tier1link

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.concurrent.TimeUnit
import javax.net.ssl.SSLContext
import javax.net.ssl.X509TrustManager

data class ClipSummary(
    val clipId: String,
    val status: String,
    val totalFrames: Int,
    val numSlots: Int,
)

data class FileEntry(
    val name: String,
    val size: Long,
    val sha256: String,
)

sealed class TlsMode {
    /** Plain HTTP, no TLS at all -- dev/back-compat path. baseUrl must use "http://". */
    object PlainHttp : TlsMode()

    /** HTTPS with a pinned fingerprint the user already has (normal, post-pairing case). */
    data class Pinned(val fingerprint: String) : TlsMode()

    /**
     * HTTPS, first-pairing-only: accepts whatever cert the server presents
     * and reports its fingerprint via [onCapture] so the caller can display
     * it for the user to save as the pin going forward. Must only be used
     * for one explicit, user-initiated pairing action -- never as a
     * standing fallback.
     */
    data class Capture(val onCapture: (String) -> Unit) : TlsMode()
}

/**
 * Kotlin port of scripts/test_tier1_link_client.py's protocol calls, against
 * src/tier1_link/server.py's four endpoints. See [TlsMode] for the three
 * connection modes this supports.
 */
class Tier1LinkClient(private val baseUrl: String, tlsMode: TlsMode = TlsMode.PlainHttp) {

    private val http: OkHttpClient = run {
        val builder = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)

        val trustManager = when (tlsMode) {
            is TlsMode.PlainHttp -> null
            is TlsMode.Pinned -> TofuTrustManager(tlsMode.fingerprint)
            is TlsMode.Capture -> TofuTrustManager(null, tlsMode.onCapture)
        }

        if (trustManager != null) {
            val sslContext = SSLContext.getInstance("TLS")
            sslContext.init(null, arrayOf<X509TrustManager>(trustManager), SecureRandom())
            builder.sslSocketFactory(sslContext.socketFactory, trustManager)
            // Fingerprint pinning is the real trust boundary here, not
            // hostname matching -- the server's self-signed cert only
            // claims "localhost" as its SAN regardless of which IP the
            // phone actually dials, so normal hostname verification would
            // always fail even against the correct, pinned server.
            builder.hostnameVerifier { _, _ -> true }
        }

        builder.build()
    }

    fun listClips(): List<ClipSummary> {
        val req = Request.Builder().url("$baseUrl/clips").build()
        http.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw java.io.IOException("GET /clips failed: ${resp.code}")
            val arr = JSONArray(resp.body!!.string())
            return (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                ClipSummary(
                    clipId = o.getString("clip_id"),
                    status = o.getString("status"),
                    totalFrames = o.optInt("total_frames", -1),
                    numSlots = o.optInt("num_slots", -1),
                )
            }
        }
    }

    fun getManifest(clipId: String): Pair<JSONObject, List<FileEntry>> {
        val req = Request.Builder().url("$baseUrl/clips/$clipId/manifest").build()
        http.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw java.io.IOException("GET manifest failed: ${resp.code}")
            val manifest = JSONObject(resp.body!!.string())
            val filesArr = manifest.getJSONArray("files")
            val files = (0 until filesArr.length()).map { i ->
                val f = filesArr.getJSONObject(i)
                FileEntry(f.getString("name"), f.getLong("size"), f.getString("sha256"))
            }
            return manifest to files
        }
    }

    /** Downloads one file to [destDir]/[name], verifying sha256+size. Returns true on success. */
    fun downloadAndVerify(clipId: String, entry: FileEntry, destDir: File): Boolean {
        return downloadAndVerify(clipId, entry, destDir, onProgress = null)
    }

    /**
     * Same as [downloadAndVerify] but streams to disk (doesn't hold the whole
     * file in memory) and reports progress via [onProgress] (bytesRead,
     * totalBytes), called from whatever thread this function runs on --
     * callers on Android should already be off the main thread (this does
     * blocking IO) and should hop back to the main thread themselves for UI
     * updates.
     */
    fun downloadAndVerify(
        clipId: String,
        entry: FileEntry,
        destDir: File,
        onProgress: ((bytesRead: Long, totalBytes: Long) -> Unit)?,
    ): Boolean {
        val req = Request.Builder().url("$baseUrl/clips/$clipId/files/${entry.name}").build()
        http.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) return false
            val body = resp.body ?: return false

            destDir.mkdirs()
            val outFile = File(destDir, entry.name)
            val digest = MessageDigest.getInstance("SHA-256")
            var bytesRead = 0L
            val buffer = ByteArray(64 * 1024)

            outFile.outputStream().use { out ->
                body.byteStream().use { input ->
                    while (true) {
                        val n = input.read(buffer)
                        if (n == -1) break
                        out.write(buffer, 0, n)
                        digest.update(buffer, 0, n)
                        bytesRead += n
                        onProgress?.invoke(bytesRead, entry.size)
                    }
                }
            }

            if (bytesRead != entry.size) {
                outFile.delete()
                return false
            }
            val actualSha = digest.digest().joinToString("") { "%02x".format(it) }
            if (actualSha != entry.sha256) {
                outFile.delete()
                return false
            }
            return true
        }
    }

    fun ackClip(clipId: String): Boolean {
        val emptyBody = ByteArray(0).toRequestBody("application/octet-stream".toMediaType())
        val req = Request.Builder()
            .url("$baseUrl/clips/$clipId/ack")
            .post(emptyBody)
            .build()
        http.newCall(req).execute().use { resp -> return resp.isSuccessful }
    }
}
