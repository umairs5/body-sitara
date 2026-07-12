package com.bodysitara.tier1link.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.bodysitara.tier1link.model.ClipPipelineState
import com.bodysitara.tier1link.model.StageStatus

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    clips: List<ClipPipelineState>,
    isBusy: Boolean,
    onRefresh: () -> Unit,
    onDiscover: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenClip: (String) -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("bodySITARA") },
                actions = {
                    IconButton(onClick = onDiscover) {
                        Icon(Icons.Filled.Search, contentDescription = "Discover on network")
                    }
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Filled.Settings, contentDescription = "Settings")
                    }
                }
            )
        },
        floatingActionButton = {
            FloatingActionButton(onClick = onRefresh) {
                Icon(Icons.Filled.Refresh, contentDescription = "Refresh clip list")
            }
        }
    ) { padding ->
        Column(modifier = Modifier.padding(padding).fillMaxSize()) {
            if (isBusy) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
            if (clips.isEmpty()) {
                Column(
                    modifier = Modifier.fillMaxSize().padding(32.dp),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        "No clips yet. Tap refresh to check the server, or search to discover it on the network.",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            } else {
                LazyColumn(modifier = Modifier.fillMaxSize()) {
                    items(clips, key = { it.clipId }) { clip ->
                        ClipRow(clip = clip, onClick = { onOpenClip(clip.clipId) })
                    }
                }
            }
        }
    }
}

@Composable
private fun ClipRow(clip: ClipPipelineState, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 6.dp)
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            StageStatusIcon(state = clip.leadStageState)
            Spacer(modifier = Modifier.width(16.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    clip.clipId.take(13) + "...",
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    clip.summary,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    "${clip.totalFrames} frames, ${clip.numSlots} slot(s)",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.outline,
                )
            }
            if (clip.pull.status == StageStatus.IN_PROGRESS) {
                CircularProgressIndicator(
                    progress = { (clip.pull.progressPercent ?: 0) / 100f },
                    modifier = Modifier.width(28.dp),
                )
            }
        }
    }
}
