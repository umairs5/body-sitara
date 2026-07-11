package com.bodysitara.tier1link

import java.security.MessageDigest
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import javax.net.ssl.X509TrustManager

/**
 * TOFU (Trust On First Use) TrustManager for the Tier1 link server's
 * self-signed cert. There's no real CA for a benchtop Pi, so this
 * deliberately does NOT do normal chain validation -- it only checks the
 * leaf cert's SHA-256 fingerprint against [expectedFingerprint], which the
 * user enters once (server prints it to console on startup; see cert.py).
 *
 * This is a simplification of the plan's "manufacturer-signed root key"
 * language (no real analog on a one-off research rig) -- it protects
 * against an impostor server AFTER pairing, not on the very first
 * connection, which is the nature of TOFU.
 *
 * If [expectedFingerprint] is null, this is "capture mode": the first
 * cert seen is accepted unconditionally and reported via [onCapture] so
 * the caller can display/save it as the pin for all subsequent
 * connections. Capture mode must only ever be used for a single, explicit,
 * user-initiated first pairing -- never as a standing "accept anything"
 * fallback, or the whole point of pinning is defeated.
 */
class TofuTrustManager(
    private val expectedFingerprint: String?,
    private val onCapture: ((String) -> Unit)? = null,
) : X509TrustManager {

    override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) {
        throw CertificateException("Client cert checking not supported (server-only TOFU pinning)")
    }

    override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {
        val leaf = chain?.firstOrNull()
            ?: throw CertificateException("No server certificate presented")
        val actual = sha256Fingerprint(leaf)

        if (expectedFingerprint == null) {
            onCapture?.invoke(actual)
            return
        }

        if (!actual.equals(expectedFingerprint, ignoreCase = true)) {
            throw CertificateException(
                "Server certificate fingerprint mismatch -- expected $expectedFingerprint, got $actual. " +
                "This could mean the server was reinstalled/re-paired, OR that a different device is " +
                "impersonating it. Re-verify the fingerprint out-of-band before trusting a new one."
            )
        }
    }

    override fun getAcceptedIssuers(): Array<X509Certificate> = arrayOf()

    companion object {
        fun sha256Fingerprint(cert: X509Certificate): String {
            val der = cert.encoded
            val digest = MessageDigest.getInstance("SHA-256").digest(der)
            return digest.joinToString("") { "%02x".format(it) }
        }
    }
}
