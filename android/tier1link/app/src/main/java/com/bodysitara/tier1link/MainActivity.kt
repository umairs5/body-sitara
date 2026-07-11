package com.bodysitara.tier1link

import android.os.Bundle
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.bodysitara.tier1link.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var clips: List<ClipSummary> = emptyList()
    private var pullInProgress = false

    private val discovery by lazy { Tier1Discovery(applicationContext) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnRefresh.setOnClickListener { refreshClips() }
        binding.btnDiscover.setOnClickListener { discoverServer() }
    }

    private fun discoverServer() {
        log("Searching for _bodysitara._tcp on the local network (mDNS)...")
        lifecycleScope.launch {
            try {
                val found = withContext(Dispatchers.IO) { discovery.discoverFirst() }
                if (found == null) {
                    log("No server found within timeout. Falling back to manual host:port entry.")
                    return@launch
                }
                log("Found: ${found.host}:${found.port} (scheme=${found.scheme}, device_id=${found.deviceId})")
                binding.editHostPort.setText("${found.host}:${found.port}")
                binding.checkUseHttps.isChecked = (found.scheme == "https")
                // A newly-discovered host may be a different physical device than
                // whatever fingerprint (if any) is currently pinned in the field --
                // clear it so buildClient() re-enters capture mode and the user
                // re-verifies, rather than silently reusing a stale/unrelated pin.
                binding.editFingerprint.setText("")
            } catch (e: Exception) {
                log("Discovery ERROR: ${e::class.simpleName}: ${e.message}")
            }
        }
    }

    private fun hostPort(): String = binding.editHostPort.text.toString().trim()

    private fun log(msg: String) {
        runOnUiThread {
            binding.textLog.append("$msg\n")
        }
    }

    /**
     * Builds a client for the current UI settings. If HTTPS is checked and
     * no fingerprint is pinned yet, this does a one-time capture-mode
     * connection first (accepting whatever cert the server presents),
     * shows the fingerprint for the user to verify out-of-band, saves it
     * into the field, and returns a properly pinned client for actual use.
     * Every subsequent call reuses the now-nonblank field, so capture mode
     * only ever fires once per pairing, not on every request.
     */
    private suspend fun buildClient(): Tier1LinkClient {
        val useHttps = binding.checkUseHttps.isChecked
        if (!useHttps) {
            return Tier1LinkClient("http://${hostPort()}", TlsMode.PlainHttp)
        }

        val existingFingerprint = binding.editFingerprint.text.toString().trim()
        if (existingFingerprint.isNotEmpty()) {
            return Tier1LinkClient("https://${hostPort()}", TlsMode.Pinned(existingFingerprint))
        }

        log("No fingerprint pinned yet -- capturing from server on this connection.")
        log("VERIFY this matches the fingerprint printed in the server's console before trusting it.")
        var captured: String? = null
        val captureClient = Tier1LinkClient(
            "https://${hostPort()}",
            TlsMode.Capture { fp -> captured = fp }
        )
        withContext(Dispatchers.IO) { captureClient.listClips() }  // triggers the handshake
        val fp = captured ?: throw IllegalStateException("Capture mode ran but no fingerprint was recorded")
        log("Captured fingerprint: $fp")
        runOnUiThread { binding.editFingerprint.setText(fp) }
        return Tier1LinkClient("https://${hostPort()}", TlsMode.Pinned(fp))
    }

    private fun refreshClips() {
        log("Listing clips from ${hostPort()} ...")
        lifecycleScope.launch {
            try {
                val client = buildClient()
                val result = withContext(Dispatchers.IO) { client.listClips() }
                clips = result
                log("Found ${result.size} clip(s)")
                renderClipList(client)
            } catch (e: Exception) {
                log("ERROR: ${e::class.simpleName}: ${e.message}")
            }
        }
    }

    private fun renderClipList(client: Tier1LinkClient) {
        val labels = clips.map { "${it.clipId.take(8)}...  [${it.status}]  frames=${it.totalFrames} slots=${it.numSlots}" }
        val adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, labels)
        binding.listClips.adapter = adapter
        binding.listClips.setOnItemClickListener { _, _, position, _ ->
            pullClip(client, clips[position])
        }
    }

    private fun pullClip(client: Tier1LinkClient, clip: ClipSummary) {
        if (pullInProgress) {
            log("A pull is already in progress -- ignoring tap.")
            return
        }
        pullInProgress = true
        log("\n=== Pulling ${clip.clipId} ===")
        lifecycleScope.launch {
            try {
                val (manifest, files) = withContext(Dispatchers.IO) { client.getManifest(clip.clipId) }
                log("Manifest OK, ${files.size} files listed")

                val destDir = File(getExternalFilesDir(null), "pulled_clips/${clip.clipId}")
                var allOk = true
                for (entry in files) {
                    val ok = withContext(Dispatchers.IO) { client.downloadAndVerify(clip.clipId, entry, destDir) }
                    allOk = allOk && ok
                    log("  ${entry.name}  ${entry.size}B  ${if (ok) "OK" else "MISMATCH"}")
                }

                if (allOk) {
                    val acked = withContext(Dispatchers.IO) { client.ackClip(clip.clipId) }
                    log(if (acked) "Acked -> ${clip.clipId}" else "Ack failed (server returned non-2xx)")
                } else {
                    log("NOT acking -- one or more files failed verification")
                }
                log("Saved under: ${destDir.absolutePath}")
            } catch (e: Exception) {
                log("ERROR: ${e::class.simpleName}: ${e.message}")
            } finally {
                pullInProgress = false
            }
        }
    }
}
