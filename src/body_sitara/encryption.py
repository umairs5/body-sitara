import os
import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization


def generate_ttp_keypair():
    # RSA-4096, matching the original SITARA paper (Section IV.A), not the
    # RSA-2048 this codebase used previously. The Pi-side cost of this
    # change is negligible -- RSA-OAEP public-key encrypt of a 16-byte AES
    # key, once per person-stream per clip, not per-frame -- while the
    # expensive private-key decrypt only ever runs on the TTP server, which
    # isn't resource-constrained. This keeps benchmark numbers directly
    # comparable to the paper's published tables.
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    return private_key, private_key.public_key()


def rsa_encrypt_key(aes_key: bytes, rsa_public_key) -> bytes:
    return rsa_public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        )
    )


def aes_gcm_encrypt(aes_key: bytes, plaintext: bytes):
    nonce = os.urandom(12)
    return nonce, AESGCM(aes_key).encrypt(nonce, plaintext, None)


def encode_crop(crop_bgr: np.ndarray, quality: int = 85) -> bytes:
    success, buf = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()
