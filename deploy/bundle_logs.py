"""Copy decision logs into the deploy build context, minus what must not ship.

The dashboard is a public demo URL, so bundling a decision log into the image
**publishes its contents**. Most of what a log holds is exactly what the demo
exists to show -- the agents' reasoning, the skips, the guardrail events, the
orders. Two fields are not:

* ``prefilter.thresholds`` -- the complete tuned calibration: delta band, DTE
  rule, every metric acceptance band. ``CLAUDE.md`` names tuned thresholds on
  the never-publish list, alongside prompts and secrets. The log records them
  deliberately, so a *private* log explains itself without ``config/``; that
  reasoning does not survive the log becoming public.
* ``prefilter.rejections`` -- per-contract failing reasons. Not secret, but it
  is the same calibration read backwards, and the dashboard never renders it.

**Neither field is displayed by any view.** Stripping them costs the demo
nothing: ``reader.py`` reads ``reason_counts`` and ``sole_reason`` for the
prefilter panel and never touches these two. Verified by grep, not by memory.

Everything else ships as written. Prompts were never in the log -- only their
sha256 -- and a check below asserts that rather than trusting it.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

# Removed from every record before the log enters the image.
STRIP = ("thresholds", "rejections")

# Distinctive strings from the operator's prompts. If any appears in a bundled
# log the bundle is refused: prompt text is the IP, and a demo URL is exactly
# where a leak would be irreversible.
PROMPT_MARKERS = (
    "You are the",
    "Return one JSON object",
    "## Hard constraints",
    "## Output",
    "narrows; it never widens",
)


def sanitise(record: dict) -> tuple[dict, list[str]]:
    payload = record.get("payload", {})
    removed = [field for field in STRIP if field in payload]
    for field in removed:
        payload.pop(field)
    return record, removed


def bundle(source: Path, target: Path, sessions: list[str] | None) -> int:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    files = sorted(source.glob("decision_log-*.jsonl"))
    if sessions:
        files = [f for f in files if any(s in f.name for s in sessions)]
    if not files:
        print(f"no decision logs matched in {source}", file=sys.stderr)
        return 1

    removed_total: collections.Counter = collections.Counter()
    leaks: list[str] = []
    written = 0

    for path in files:
        out_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue          # a half-written final line is not an error
            record, removed = sanitise(record)
            for field in removed:
                removed_total[field] += 1
            text = json.dumps(record, separators=(",", ":"))
            for marker in PROMPT_MARKERS:
                if marker in text:
                    leaks.append(f"{path.name}: {marker!r}")
            out_lines.append(text)

        (target / path.name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        written += len(out_lines)
        print(f"  {path.name}: {len(out_lines)} records")

    print()
    print(f"bundled {written} records from {len(files)} session(s) into {target}")
    for field, n in removed_total.most_common():
        print(f"  stripped {field}: {n} occurrence(s)")

    if leaks:
        print("\nREFUSING TO BUNDLE -- prompt text found:", file=sys.stderr)
        for leak in leaks[:10]:
            print(f"  {leak}", file=sys.stderr)
        shutil.rmtree(target)
        return 2

    print("  prompt-text scan: clean")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m deploy.bundle_logs")
    p.add_argument("--source", type=Path, default=Path("logs"))
    p.add_argument("--target", type=Path, default=Path("deploy/logs"))
    p.add_argument("--sessions", nargs="*", default=None,
                   help="session dates to include; default all")
    args = p.parse_args(argv)
    return bundle(args.source, args.target, args.sessions)


if __name__ == "__main__":
    raise SystemExit(main())
