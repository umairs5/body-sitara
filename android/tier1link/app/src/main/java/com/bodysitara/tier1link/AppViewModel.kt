package com.bodysitara.tier1link

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.bodysitara.tier1link.model.ClipPipelineState
import com.bodysitara.tier1link.model.StageState
import com.bodysitara.tier1link.model.StageStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class ConnectionSettings(
    val hostPort: String = "10.130.155.182:8443",
    val useHttps: Boolean = true,
    val pinnedFingerprint: String = "",
)

data class AppUiState(
    val connection: ConnectionSettings = ConnectionSettings(),
    val clips: List<ClipPipelineState> = emptyList(),
    val logLines: List<String> = emptyList(),
    val isBusy: Boolean = false,
)

class AppViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(AppUiState())
    val uiState: StateFlow<AppUiState> = _uiState

    private val discovery by lazy { Tier1Discovery(getApplication<Application>().applicationContext) }

    fun updateConnection(update: (ConnectionSettings) -> ConnectionSettings) {
        _uiState.update { it.copy(connection = update(it.connection)) }
    }

    private fun log(msg: String) {
        _uiState.update { it.copy(logLines = it.logLines + msg) }
    }

    fun discoverServer() {
        log("Searching for _bodysitara._tcp on the local network (mDNS)...")
        viewModelScope.launch {
            _uiState.update { it.copy(isBusy = true) }
            try {
                val found = withContext(Dispatchers.IO) { discovery.discoverFirst() }
                if (found == null) {
                    log("No server found within timeout.")
                } else {
                    log("Found: ${found.host}:${found.port} (scheme=${found.scheme})")
                    updateConnection {
                        it.copy(
                            hostPort = "${found.host}:${found.port}",
                            useHttps = found.scheme == "https",
                            pinnedFingerprint = "",  // new host -> re-verify, don't reuse a stale pin
                        )
                    }
                }
            } catch (e: Exception) {
                log("Discovery ERROR: ${e::class.simpleName}: ${e.message}")
            } finally {
                _uiState.update { it.copy(isBusy = false) }
            }
        }
    }

    private suspend fun buildClient(): Tier1LinkClient {
        val conn = _uiState.value.connection
        if (!conn.useHttps) {
            return Tier1LinkClient("http://${conn.hostPort}", TlsMode.PlainHttp)
        }
        if (conn.pinnedFingerprint.isNotBlank()) {
            return Tier1LinkClient("https://${conn.hostPort}", TlsMode.Pinned(conn.pinnedFingerprint))
        }

        log("No fingerprint pinned -- capturing from server on this connection. VERIFY it out-of-band.")
        var captured: String? = null
        val captureClient = Tier1LinkClient(
            "https://${conn.hostPort}",
            TlsMode.Capture { fp -> captured = fp }
        )
        withContext(Dispatchers.IO) { captureClient.listClips() }
        val fp = captured ?: throw IllegalStateException("Capture mode ran but no fingerprint was recorded")
        log("Captured fingerprint: $fp")
        updateConnection { it.copy(pinnedFingerprint = fp) }
        return Tier1LinkClient("https://${conn.hostPort}", TlsMode.Pinned(fp))
    }

    fun refreshClips() {
        viewModelScope.launch {
            _uiState.update { it.copy(isBusy = true) }
            try {
                val client = buildClient()
                val summaries = withContext(Dispatchers.IO) { client.listClips() }
                log("Found ${summaries.size} clip(s)")
                _uiState.update { state ->
                    val existingById = state.clips.associateBy { it.clipId }
                    val merged = summaries.map { s ->
                        existingById[s.clipId] ?: ClipPipelineState(
                            clipId = s.clipId,
                            totalFrames = s.totalFrames,
                            numSlots = s.numSlots,
                        )
                    }
                    state.copy(clips = merged)
                }
            } catch (e: Exception) {
                log("ERROR: ${e::class.simpleName}: ${e.message}")
            } finally {
                _uiState.update { it.copy(isBusy = false) }
            }
        }
    }

    private fun updateClip(clipId: String, transform: (ClipPipelineState) -> ClipPipelineState) {
        _uiState.update { state ->
            state.copy(clips = state.clips.map { if (it.clipId == clipId) transform(it) else it })
        }
    }

    /** Runs the real Pull stage: manifest -> download all files (with progress) -> ack. */
    fun pullClip(clipId: String) {
        viewModelScope.launch {
            updateClip(clipId) { it.copy(pull = StageState(status = StageStatus.IN_PROGRESS, progressPercent = 0)) }
            try {
                val client = buildClient()
                val (_, files) = withContext(Dispatchers.IO) { client.getManifest(clipId) }

                val destDir = File(getApplication<Application>().getExternalFilesDir(null), "pulled_clips/$clipId")
                val totalBytes = files.sumOf { it.size }.coerceAtLeast(1)
                var bytesDoneSoFar = 0L
                var allOk = true

                for (entry in files) {
                    val fileStartBytes = bytesDoneSoFar
                    val ok = withContext(Dispatchers.IO) {
                        client.downloadAndVerify(clipId, entry, destDir) { read, _ ->
                            val overallPercent = (((fileStartBytes + read) * 100) / totalBytes).toInt()
                            updateClip(clipId) {
                                it.copy(pull = it.pull.copy(progressPercent = overallPercent.coerceIn(0, 100)))
                            }
                        }
                    }
                    allOk = allOk && ok
                    bytesDoneSoFar += entry.size
                    if (!ok) log("  $clipId: ${entry.name} FAILED verification")
                }

                if (allOk) {
                    withContext(Dispatchers.IO) { client.ackClip(clipId) }
                    val videoPath = File(destDir, "output_rtm.mp4").takeIf { it.exists() }?.absolutePath
                    updateClip(clipId) {
                        it.copy(
                            pull = StageState(status = StageStatus.DONE, progressPercent = 100),
                            incomingVideoPath = videoPath,
                        )
                    }
                    log("Pull complete: $clipId")
                } else {
                    updateClip(clipId) { it.copy(pull = it.pull.copy(status = StageStatus.FAILED)) }
                }
            } catch (e: Exception) {
                log("Pull ERROR for $clipId: ${e::class.simpleName}: ${e.message}")
                updateClip(clipId) { it.copy(pull = it.pull.copy(status = StageStatus.FAILED, message = e.message)) }
            }
        }
    }

    // No real implementation exists yet for these two -- see model/ClipPipeline.kt's
    // doc comment. Left as explicit no-ops (not wired to any button) rather than
    // faked, so the UI's "not connected" state is honest about current capability.
}
