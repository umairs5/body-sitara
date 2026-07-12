package com.bodysitara.tier1link.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.bodysitara.tier1link.model.StageState
import com.bodysitara.tier1link.model.StageStatus

/** A small icon summarizing a single stage's status -- used in both the clip-list
 * rows and the detail-screen stepper. */
@Composable
fun StageStatusIcon(state: StageState, modifier: Modifier = Modifier) {
    Box(modifier = modifier.size(24.dp)) {
        when (state.status) {
            StageStatus.DONE -> Icon(
                Icons.Filled.CheckCircle, contentDescription = "Done",
                tint = MaterialTheme.colorScheme.primary,
            )
            StageStatus.FAILED -> Icon(
                Icons.Filled.Error, contentDescription = "Failed",
                tint = MaterialTheme.colorScheme.error,
            )
            StageStatus.IN_PROGRESS -> {
                if (state.progressPercent != null) {
                    CircularProgressIndicator(
                        progress = { state.progressPercent / 100f },
                        strokeWidth = 2.5.dp,
                    )
                } else {
                    CircularProgressIndicator(strokeWidth = 2.5.dp)
                }
            }
            StageStatus.NOT_CONNECTED -> Icon(
                Icons.Filled.RadioButtonUnchecked, contentDescription = "Not connected",
                tint = MaterialTheme.colorScheme.outlineVariant,
            )
            StageStatus.PENDING -> Icon(
                Icons.Filled.RadioButtonUnchecked, contentDescription = "Pending",
                tint = MaterialTheme.colorScheme.outline,
            )
        }
    }
}

fun stageStatusColor(status: StageStatus, defaultColor: Color, errorColor: Color): Color = when (status) {
    StageStatus.FAILED -> errorColor
    StageStatus.NOT_CONNECTED -> defaultColor.copy(alpha = 0.4f)
    else -> defaultColor
}
