package com.bodysitara.tier1link.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.bodysitara.tier1link.ConnectionSettings

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    connection: ConnectionSettings,
    onBack: () -> Unit,
    onUpdate: ((ConnectionSettings) -> ConnectionSettings) -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Connection Settings") },
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
            OutlinedTextField(
                value = connection.hostPort,
                onValueChange = { v -> onUpdate { it.copy(hostPort = v) } },
                label = { Text("Server host:port") },
                placeholder = { Text("e.g. 10.130.155.182:8443") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            Spacer(modifier = Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(
                    checked = connection.useHttps,
                    onCheckedChange = { v -> onUpdate { it.copy(useHttps = v) } },
                )
                Text("Use HTTPS (TOFU pinned)")
            }
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                "Server fingerprint (SHA-256, printed on server startup)",
                style = MaterialTheme.typography.labelLarge,
            )
            Spacer(modifier = Modifier.height(4.dp))
            OutlinedTextField(
                value = connection.pinnedFingerprint,
                onValueChange = { v -> onUpdate { it.copy(pinnedFingerprint = v) } },
                placeholder = { Text("blank = capture on first connect") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "Leaving this blank connects once in \"capture\" mode, shows you the server's " +
                    "certificate fingerprint, and pins it here for you. Verify it against the " +
                    "server's own console output before trusting it.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
