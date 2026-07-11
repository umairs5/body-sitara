package com.bodysitara.tier1link

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

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

/**
 * Kotlin port of scripts/test_tier1_link_client.py's protocol calls, against
 * src/tier1_link/server.py's four endpoints. Plain HTTP + explicit base URL
 * for now -- no mDNS discovery or TLS pinning yet (see plan section 2.1,
 * phased: plain IP first, then TOFU-pinned HTTPS + mDNS once this baseline
 * flow is proven).
 */
class Tier1LinkClient(private val baseUrl: String) {

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

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
        val req = Request.Builder().url("$baseUrl/clips/$clipId/files/${entry.name}").build()
        http.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) return false
            val bytes = resp.body!!.bytes()
            if (bytes.size.toLong() != entry.size) return false

            val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
            val actualSha = digest.joinToString("") { "%02x".format(it) }
            if (actualSha != entry.sha256) return false

            destDir.mkdirs()
            File(destDir, entry.name).writeBytes(bytes)
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
