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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnRefresh.setOnClickListener { refreshClips() }
    }

    private fun baseUrl(): String {
        val hostPort = binding.editHostPort.text.toString().trim()
        return "http://$hostPort"
    }

    private fun log(msg: String) {
        runOnUiThread {
            binding.textLog.append("$msg\n")
        }
    }

    private fun refreshClips() {
        val client = Tier1LinkClient(baseUrl())
        log("Listing clips from ${baseUrl()} ...")
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) { client.listClips() }
                clips = result
                log("Found ${result.size} clip(s)")
                renderClipList(client)
            } catch (e: Exception) {
                log("ERROR: ${e.message}")
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
