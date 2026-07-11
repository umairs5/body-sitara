package com.bodysitara.tier1link

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.net.wifi.WifiManager
import android.util.Log
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

private const val TAG = "Tier1Discovery"
const val SERVICE_TYPE = "_bodysitara._tcp."

data class DiscoveredServer(
    val host: String,
    val port: Int,
    val deviceId: String?,
    val scheme: String,
)

/**
 * Android NsdManager-based mDNS discovery for the Tier1 link server (see
 * src/tier1_link/discovery.py -- same _bodysitara._tcp service type, same
 * TXT record keys). Requires a multicast lock: many WiFi chipsets drop
 * multicast UDP (which mDNS depends on) by default to save power.
 */
class Tier1Discovery(private val context: Context) {

    private val nsdManager = context.getSystemService(Context.NSD_SERVICE) as NsdManager
    private var multicastLock: WifiManager.MulticastLock? = null

    private fun acquireMulticastLock() {
        val wifi = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        val lock = wifi.createMulticastLock("tier1link-mdns")
        lock.setReferenceCounted(true)
        lock.acquire()
        multicastLock = lock
        Log.d(TAG, "Multicast lock acquired: held=${lock.isHeld}")
    }

    private fun releaseMulticastLock() {
        multicastLock?.let { if (it.isHeld) it.release() }
        multicastLock = null
    }

    /**
     * Discovers the first available _bodysitara._tcp service within
     * [timeoutMs]. Returns null if nothing was found in time. Only ever
     * resumes the continuation once (guarded by [resolved]) -- NsdManager's
     * callbacks can otherwise fire multiple times per discovery session.
     */
    suspend fun discoverFirst(timeoutMs: Long = 8000): DiscoveredServer? {
        acquireMulticastLock()
        return try {
            suspendCancellableCoroutine { cont ->
                var resolved = false

                val discoveryListener = object : NsdManager.DiscoveryListener {
                    override fun onDiscoveryStarted(serviceType: String) {
                        Log.d(TAG, "onDiscoveryStarted: $serviceType")
                    }
                    override fun onServiceFound(service: NsdServiceInfo) {
                        Log.d(TAG, "onServiceFound: ${service.serviceName} type=${service.serviceType}")
                        if (resolved) return
                        nsdManager.resolveService(service, object : NsdManager.ResolveListener {
                            override fun onResolveFailed(info: NsdServiceInfo, errorCode: Int) {
                                Log.e(TAG, "onResolveFailed: ${info.serviceName} errorCode=$errorCode")
                            }
                            override fun onServiceResolved(info: NsdServiceInfo) {
                                Log.d(TAG, "onServiceResolved: ${info.serviceName} host=${info.host} port=${info.port}")
                                if (resolved) return
                                resolved = true
                                val deviceId = info.attributes["device_id"]?.let { String(it) }
                                val scheme = info.attributes["scheme"]?.let { String(it) } ?: "https"
                                val result = DiscoveredServer(
                                    host = info.host.hostAddress ?: return,
                                    port = info.port,
                                    deviceId = deviceId,
                                    scheme = scheme,
                                )
                                if (cont.isActive) cont.resume(result)
                            }
                        })
                    }
                    override fun onServiceLost(service: NsdServiceInfo) {
                        Log.d(TAG, "onServiceLost: ${service.serviceName}")
                    }
                    override fun onDiscoveryStopped(serviceType: String) {
                        Log.d(TAG, "onDiscoveryStopped: $serviceType")
                    }
                    override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) {
                        Log.e(TAG, "onStartDiscoveryFailed: $serviceType errorCode=$errorCode")
                        if (!resolved && cont.isActive) {
                            cont.resumeWithException(RuntimeException("mDNS discovery start failed: $errorCode"))
                        }
                    }
                    override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) {
                        Log.e(TAG, "onStopDiscoveryFailed: $serviceType errorCode=$errorCode")
                    }
                }

                cont.invokeOnCancellation {
                    runCatching { nsdManager.stopServiceDiscovery(discoveryListener) }
                }

                Log.d(TAG, "Calling discoverServices for $SERVICE_TYPE")
                nsdManager.discoverServices(SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, discoveryListener)

                // Manual timeout: stop discovery and resume with null if nothing found.
                android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                    if (!resolved) {
                        Log.d(TAG, "Timeout reached with no resolved service.")
                        runCatching { nsdManager.stopServiceDiscovery(discoveryListener) }
                        if (cont.isActive) cont.resume(null)
                    } else {
                        runCatching { nsdManager.stopServiceDiscovery(discoveryListener) }
                    }
                }, timeoutMs)
            }
        } finally {
            releaseMulticastLock()
        }
    }
}
