plugins {
    id("com.android.application") version "8.7.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
}

// Build outputs (build/) get rewritten constantly by Gradle/Android Studio;
// inside a OneDrive-synced folder this causes intermittent "unable to
// delete directory" failures when OneDrive has a file open mid-sync.
// Redirect all build dirs to a local, non-synced path instead.
val localBuildRoot = File(System.getenv("LOCALAPPDATA") ?: System.getProperty("user.home"), "tier1link-gradle-build")
rootProject.layout.buildDirectory.set(File(localBuildRoot, "root"))
subprojects {
    layout.buildDirectory.set(File(localBuildRoot, name))
}
