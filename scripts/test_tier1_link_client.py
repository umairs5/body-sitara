"""
Minimal test client for the Tier1 link server (src/tier1_link/server.py).

Simulates what the real Android client will do: list clips, pull a
manifest, download every file the manifest lists, verify each against
its sha256, then ack the clip. Proves the server-side protocol works
end-to-end before investing in a real Android app shell.

Usage:
    python scripts/test_tier1_link_client.py --host 127.0.0.1 --port 8000 --out scratch/pulled_clips
"""
import argparse
import hashlib
import json
import os
import urllib.request


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="scratch/pulled_clips")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"

    print(f"Discovering clips from {base} ...")
    clips = json.loads(urllib.request.urlopen(f"{base}/clips").read())
    if not clips:
        print("No clips available.")
        return
    for c in clips:
        print(f"  {c['clip_id']}  status={c['status']}  frames={c['total_frames']}  slots={c['num_slots']}")

    for c in clips:
        clip_id = c["clip_id"]
        if c["status"] == "acknowledged":
            print(f"\nSkipping already-acknowledged clip {clip_id}")
            continue

        print(f"\n=== Pulling clip {clip_id} ===")
        manifest = json.loads(urllib.request.urlopen(f"{base}/clips/{clip_id}/manifest").read())
        clip_out_dir = os.path.join(args.out, clip_id)
        os.makedirs(clip_out_dir, exist_ok=True)

        with open(os.path.join(clip_out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        all_ok = True
        for finfo in manifest["files"]:
            name, expected_sha, expected_size = finfo["name"], finfo["sha256"], finfo["size"]
            url = f"{base}/clips/{clip_id}/files/{name}"
            data = urllib.request.urlopen(url).read()
            actual_sha = sha256_of_bytes(data)
            ok = (actual_sha == expected_sha) and (len(data) == expected_size)
            all_ok &= ok
            status = "OK" if ok else "MISMATCH"
            print(f"  {name:<24} {len(data):>10} bytes  sha256={'match' if ok else 'MISMATCH'}  [{status}]")
            if ok:
                with open(os.path.join(clip_out_dir, name), "wb") as f:
                    f.write(data)

        if all_ok:
            req = urllib.request.Request(f"{base}/clips/{clip_id}/ack", method="POST")
            ack_resp = json.loads(urllib.request.urlopen(req).read())
            print(f"  Acked -> {ack_resp}")
        else:
            print("  NOT acking -- one or more files failed checksum verification.")

    print(f"\nDone. Pulled clips saved under: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
