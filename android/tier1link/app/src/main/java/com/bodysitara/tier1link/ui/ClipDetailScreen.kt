package com.bodysitara.tier1link.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CloudQueue
import androidx.compose.material.icons.filled.Movie
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.bodysitara.tier1link.model.ClipPipelineState
import com.bodysitara.tier1link.model.StageState
import com.bodysitara.tier1link.model.StageStatus

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ClipDetailScreen(
    clip: ClipPipelineState,
    onBack: () -> Unit,
    onPull: () -> Unit,
    onRunBackgroundFill: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(clip.clipId.take(13) + "...") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp)
        ) {
            Text(
                "${clip.totalFrames} frames · ${clip.numSlots} person slot(s)",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(modifier = Modifier.height(16.dp))

            StageCard(
                title = "1. Pull from Pi",
                subtitle = "Download the de-identified signal bundle over the local Tier1 link.",
                state = clip.pull,
            ) {
                if (clip.pull.status == StageStatus.PENDING || clip.pull.status == StageStatus.FAILED) {
                    Button(onClick = onPull) { Text(if (clip.pull.status == StageStatus.FAILED) "Retry pull" else "Pull clip") }
                }
                if (clip.incomingVideoPath != null) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("Incoming (anonymized) video:", style = MaterialTheme.typography.labelMedium)
                    Spacer(modifier = Modifier.height(4.dp))
                    VideoPlayer(filePath = clip.incomingVideoPath)
                }
            }

            StepConnector()

            Row(modifier = Modifier.fillMaxWidth()) {
                StageCard(
                    title = "2a. Background fill",
                    subtitle = "On-phone: LaMa-Dilated (ONNX) inpaints the person-shaped hole.",
                    state = clip.backgroundFill,
                    modifier = Modifier.weight(1f),
                ) {
                    if (clip.backgroundFill.status == StageStatus.PENDING || clip.backgroundFill.status == StageStatus.FAILED) {
                        Button(onClick = onRunBackgroundFill) {
                            Text(if (clip.backgroundFill.status == StageStatus.FAILED) "Retry fill" else "Fill background")
                        }
                    }
                    if (clip.filledVideoPath != null) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("Filled background video:", style = MaterialTheme.typography.labelMedium)
                        Spacer(modifier = Modifier.height(4.dp))
                        VideoPlayer(filePath = clip.filledVideoPath)
                    }
                }
                Spacer(modifier = Modifier.width(8.dp))
                StageCard(
                    title = "2b. Cloud generate",
                    subtitle = "RunPod: WanVideo generates the synthetic avatar.",
                    state = clip.cloudGenerate,
                    icon = Icons.Filled.CloudQueue,
                    modifier = Modifier.weight(1f),
                ) {
                    if (clip.cloudGenerate.status == StageStatus.NOT_CONNECTED) {
                        NotConnectedNote("RunPod gateway isn't implemented yet (Phase 2).")
                    }
                }
            }

            StepConnector()

            StageCard(
                title = "3. Composite",
                subtitle = "On-phone: composite the avatar onto the filled background, re-mux audio.",
                state = clip.composite,
                icon = Icons.Filled.Movie,
            ) {
                if (clip.composite.status == StageStatus.NOT_CONNECTED) {
                    NotConnectedNote("On-device compositor isn't implemented yet (Phase 3).")
                }
                if (clip.finalVideoPath != null) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("Final anonymized video:", style = MaterialTheme.typography.labelMedium)
                    Spacer(modifier = Modifier.height(4.dp))
                    VideoPlayer(filePath = clip.finalVideoPath)
                }
            }
        }
    }
}

@Composable
private fun StepConnector() {
    Row(
        modifier = Modifier.fillMaxWidth().height(24.dp),
        horizontalArrangement = Arrangement.Center,
    ) {
        HorizontalDivider(modifier = Modifier.width(2.dp).fillMaxSize())
    }
}

@Composable
private fun NotConnectedNote(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.outline,
    )
}

@Composable
private fun StageCard(
    title: String,
    subtitle: String,
    state: StageState,
    modifier: Modifier = Modifier,
    icon: androidx.compose.ui.graphics.vector.ImageVector? = null,
    content: @Composable () -> Unit = {},
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (state.status == StageStatus.NOT_CONNECTED)
                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f)
            else MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                StageStatusIcon(state = state)
                Spacer(modifier = Modifier.width(8.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    Text(subtitle, style = MaterialTheme.typography.bodySmall)
                }
            }
            if (state.status == StageStatus.IN_PROGRESS) {
                Spacer(modifier = Modifier.height(8.dp))
                if (state.progressPercent != null) {
                    LinearProgressIndicator(
                        progress = { state.progressPercent / 100f },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Text("${state.progressPercent}%", style = MaterialTheme.typography.labelSmall)
                } else {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
            }
            state.message?.let {
                Spacer(modifier = Modifier.height(4.dp))
                val messageColor = if (state.status == StageStatus.FAILED)
                    MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant
                Text(it, style = MaterialTheme.typography.bodySmall, color = messageColor)
            }
            Spacer(modifier = Modifier.height(8.dp))
            content()
        }
    }
}
