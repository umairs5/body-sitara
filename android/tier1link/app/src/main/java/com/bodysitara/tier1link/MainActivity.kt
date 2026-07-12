package com.bodysitara.tier1link

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.bodysitara.tier1link.ui.ClipDetailScreen
import com.bodysitara.tier1link.ui.HomeScreen
import com.bodysitara.tier1link.ui.SettingsScreen
import com.bodysitara.tier1link.ui.theme.Tier1LinkTheme

class MainActivity : ComponentActivity() {

    private val viewModel: AppViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Tier1LinkTheme {
                val navController = rememberNavController()
                val state by viewModel.uiState.collectAsState()

                NavHost(navController = navController, startDestination = "home") {
                    composable("home") {
                        HomeScreen(
                            clips = state.clips,
                            isBusy = state.isBusy,
                            onRefresh = { viewModel.refreshClips() },
                            onDiscover = { viewModel.discoverServer() },
                            onOpenSettings = { navController.navigate("settings") },
                            onOpenClip = { clipId -> navController.navigate("clip/$clipId") },
                        )
                    }
                    composable(
                        "clip/{clipId}",
                        arguments = listOf(navArgument("clipId") { type = NavType.StringType }),
                    ) { backStackEntry ->
                        val clipId = backStackEntry.arguments?.getString("clipId") ?: return@composable
                        val clip = state.clips.firstOrNull { it.clipId == clipId }
                        if (clip != null) {
                            ClipDetailScreen(
                                clip = clip,
                                onBack = { navController.popBackStack() },
                                onPull = { viewModel.pullClip(clipId) },
                                onRunBackgroundFill = { viewModel.runBackgroundFill(clipId) },
                            )
                        }
                    }
                    composable("settings") {
                        SettingsScreen(
                            connection = state.connection,
                            onBack = { navController.popBackStack() },
                            onUpdate = { update -> viewModel.updateConnection(update) },
                        )
                    }
                }
            }
        }
    }
}
