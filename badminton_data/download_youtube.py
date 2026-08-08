#!/usr/bin/env python3
"""Download the BFMD source videos from YouTube (BWF TV) via yt-dlp.

Reads Badminton_video_list.csv (match_name, type, youtube_url, ...) and saves
each video to the location the annotations expect:
    singles -> videos/<tournament>/<match_name>.mp4
    doubles -> videos_doubles/<match_name>.mp4

All annotations index frames in 0-based decode order on a 1280x720 / 30 fps
stream, so videos are downloaded capped at 720p and merged into mp4.

Examples
  python download_youtube.py --list              # show what would be downloaded
  python download_youtube.py                     # download everything missing
  python download_youtube.py --match Antonsen    # only matches containing a string
  python download_youtube.py --force             # re-download even if file exists

Requires: pip install yt-dlp   (and ffmpeg on PATH for stream merging)
"""
import argparse
import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LIST = os.path.join(ROOT, "Badminton_video_list.csv")
FMT = "bv*[height<=720]+ba/b[height<=720]"


def load_rows(match_filter):
    with open(LIST, newline="") as f:
        rows = list(csv.DictReader(f))
    if match_filter:
        rows = [r for r in rows if match_filter.lower() in r["match_name"].lower()]
    return rows


def target_path(row):
    name = row["match_name"]
    if row["type"] == "doubles":
        return os.path.join(ROOT, "videos_doubles", name + ".mp4")
    vdir = os.path.join(ROOT, "videos")
    if os.path.isdir(vdir):
        for tour in sorted(os.listdir(vdir)):
            if os.path.isdir(os.path.join(vdir, tour)) and name.startswith(tour):
                return os.path.join(vdir, tour, name + ".mp4")
    m = re.match(r"^(.+?-(?:19|20)\d\d)-", name)   # tournament = prefix up to the year
    if m:
        return os.path.join(vdir, m.group(1), name + ".mp4")
    return os.path.join(vdir, name + ".mp4")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", help="only rows whose match_name contains this substring")
    ap.add_argument("--list", action="store_true", help="print the plan, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    rows = load_rows(args.match)
    if not rows:
        sys.exit("no rows selected — check --match / Badminton_video_list.csv")

    todo = []
    for r in rows:
        dst = target_path(r)
        exists = os.path.exists(dst)          # true for real files and live symlinks
        state = "have" if exists and not args.force else "get "
        if args.list:
            print(f"[{state}] {r['status']:>10}  {r['youtube_url']:45}  ->  "
                  f"{os.path.relpath(dst, ROOT)}")
        if state == "get " and r["youtube_url"]:
            todo.append((r, dst))
    if args.list:
        return

    for i, (r, dst) in enumerate(todo, 1):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"\n=== [{i}/{len(todo)}] {r['match_name']}")
        cmd = ["yt-dlp", "-f", FMT, "--merge-output-format", "mp4",
               "-o", dst, r["youtube_url"]]
        if subprocess.run(cmd).returncode != 0:
            print(f"FAILED: {r['match_name']}", file=sys.stderr)
    print(f"\ndone ({len(todo)} downloaded)" if todo else "nothing to download")


if __name__ == "__main__":
    main()
