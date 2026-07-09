
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from body_sitara.pipeline import process_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bodySITARA RTMPose pipeline")

    parser.add_argument("input", type=str, help="Input video path")
    parser.add_argument("--output", type=str, default="/tmp/output_rtm.mp4",
                        help="Output video path (ignored in benchmark mode)")
    parser.add_argument("--enc-dir", type=str, default="data/output/encrypted",
                        help="Encrypted output directory")
    parser.add_argument("--skip-n", type=int, default=5,
                        help="Run full inference every N frames. Ignored if --movement-adaptive is set.")
    parser.add_argument("--movement-adaptive", action="store_true",
                        help="Enable movement-adaptive skip (slow/medium/fast tiers).")
    parser.add_argument("--benchmark", action="store_true",
                        help="Disable video save, crypto, and drawing. Measures pure pipeline latency.")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Append timing summary row to this CSV file.")
    parser.add_argument("--headless", action="store_true",
                        help="Disable display window (SSH/Pi mode).")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip VideoWriter.")
    parser.add_argument("--no-blur", action="store_true",
                        help="Disable anonymization.")
    parser.add_argument("--anonymizer", type=str, default="convexhull",
                        choices=["convexhull", "selfie_seg0", "selfie_seg1"],
                        help="Anonymization backend: convexhull (default), "
                             "selfie_seg0 (MediaPipe general), selfie_seg1 (MediaPipe landscape).")

    args = parser.parse_args()

    process_video(
        input_path        = args.input,
        output_path       = args.output,
        blur_bodies       = not args.no_blur,
        enc_output_dir    = args.enc_dir,
        headless          = args.headless,
        save_video        = not args.no_save,
        benchmark         = args.benchmark,
        skip_n            = args.skip_n,
        movement_adaptive = args.movement_adaptive,
        csv_out           = args.csv_out,
        anonymizer        = args.anonymizer,
    )
