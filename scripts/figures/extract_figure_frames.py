"""Extract representative frames from MAD-eval `.avi` clips for each curated
figure example.

Reads `figure_examples/_materialize_summary.json` produced by
`materialize_figure_examples.py`. For every example, writes 3 JPEG frames per
context clip and (if available) per target clip to that example's `frames/`
directory:

    frames/context_<i>__<clip_id>__t0.jpg          (start)
    frames/context_<i>__<clip_id>__tmid.jpg        (midpoint)
    frames/context_<i>__<clip_id>__tend.jpg        (last frame ~clip_dur−0.05)
    frames/target__<clip_id>__t0.jpg / __tmid.jpg / __tend.jpg

The script is idempotent: if a frame already exists, it is skipped. Raw video
files are read-only; only `.jpg` outputs are written.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/jumbo/jinlab/Sally/ForSeeBench")
SUMMARY = ROOT / "figure_examples/_materialize_summary.json"
FFMPEG = shutil.which("ffmpeg") or "/thayerfs/apps/tecplot_360_2012_R1/bin/ffmpeg"


def probe_duration(path: Path) -> float | None:
    """Use ffmpeg to read clip duration in seconds. Returns None on failure."""
    try:
        out = subprocess.run(
            [FFMPEG, "-i", str(path)],
            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
            check=False, text=True, timeout=20,
        )
    except Exception:
        return None
    text = out.stderr or ""
    # Look for: Duration: 00:00:04.27,
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
    if not m:
        return None
    h, m_, s = m.groups()
    return int(h) * 3600 + int(m_) * 60 + float(s)


def extract_frame(video: Path, ts: float, out_path: Path) -> bool:
    """Run ffmpeg to extract a single frame at `ts` seconds. Idempotent."""
    if out_path.exists():
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The system ffmpeg here is an old SVN build and uses `-vframes` rather
    # than the modern `-frames:v` option. Output one JPEG via mjpeg.
    cmd = [
        FFMPEG, "-y",
        "-ss", f"{ts:.3f}", "-i", str(video),
        "-vframes", "1",
        "-f", "image2",
        "-vcodec", "mjpeg",
        "-qscale", "3",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg failed for {video.name} @ {ts:.2f}s: {e.stderr.decode()[:120]}",
              file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"  ffmpeg timeout for {video.name}", file=sys.stderr)
        return False
    return out_path.exists()


def clip_id_from_path(p: str) -> str:
    """Strip the directory + .avi suffix to get a stable token for filenames."""
    name = Path(p).stem
    return name.replace(".", "p")


def process_clip(video_path_str: str | None, role: str, role_idx: int,
                 frames_dir: Path) -> int:
    if not video_path_str:
        return 0
    p = ROOT / video_path_str if not video_path_str.startswith("/") else Path(video_path_str)
    if not p.exists():
        print(f"  missing video: {p}", file=sys.stderr)
        return 0
    duration = probe_duration(p)
    if duration is None or duration <= 0:
        print(f"  could not probe duration: {p.name}", file=sys.stderr)
        return 0
    cid = clip_id_from_path(video_path_str)
    base = f"{role}_{role_idx:02d}__{cid}" if role != "target" else f"target__{cid}"
    timestamps = []
    timestamps.append(("t0", 0.0))
    if duration > 0.5:
        timestamps.append(("tmid", duration / 2.0))
    if duration > 0.3:
        timestamps.append(("tend", max(0.0, duration - 0.08)))
    written = 0
    for tag, ts in timestamps:
        out = frames_dir / f"{base}__{tag}.jpg"
        if extract_frame(p, ts, out):
            written += 1
    return written


def main() -> int:
    summary = json.loads(SUMMARY.read_text())
    total = 0
    bucket_dirs = {
        "good": ROOT / "figure_examples/good_examples",
        "bad": ROOT / "figure_examples/bad_examples",
        "baseline_comparison": ROOT / "figure_examples/baseline_comparison_examples",
    }
    for bucket, items in summary.items():
        if bucket not in bucket_dirs:
            continue
        for item in items:
            edir = bucket_dirs[bucket] / item["example"] / "frames"
            edir.mkdir(parents=True, exist_ok=True)
            n = 0
            ctx_paths = item.get("ctx_paths") or []
            for i, cpath in enumerate(ctx_paths):
                n += process_clip(cpath, "context", i + 1, edir)
            n += process_clip(item.get("target_path"), "target", 0, edir)
            print(f"{bucket} {item['example']}: extracted {n} frames", file=sys.stderr)
            total += n
            # Mark missing if nothing extracted
            extracted_files = list(edir.glob("*.jpg"))
            if not extracted_files:
                (edir.parent / "frames_missing.txt").write_text(
                    "Could not extract any frames — clips may be missing or unsupported.\n"
                )
            else:
                # Clean up stale missing marker if frames now exist.
                stale = edir.parent / "frames_missing.txt"
                if stale.exists():
                    stale.unlink()
    print(f"Total frames written: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
