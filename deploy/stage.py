"""Assemble the deploy staging directory, OUTSIDE the repository.

Outside is the whole point. The first deploy built from a directory inside the
repo and silently inherited the repo's ``.dockerignore``, whose ``logs/`` deny
rule dropped the sessions -- the upload was 16 kB and the build failed on a
missing COPY. Staging outside removes every such interaction: what is in the
directory is exactly what ships, and no parent file can add to it or take from
it.

    python -m deploy.bundle_logs            # sanitise logs -> deploy/logs
    python -m deploy.stage --out <dir>      # assemble the build context
    cd <dir> && railway up --service deepsees-dashboard
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Everything the image needs, and nothing else. An explicit list rather than a
# copy-and-exclude, so a new directory in src/ cannot ship by accident.
FILES = {
    "src/__init__.py": "src/__init__.py",
    "src/dashboard/__init__.py": "src/dashboard/__init__.py",
    "src/dashboard/app.py": "src/dashboard/app.py",
    "src/dashboard/reader.py": "src/dashboard/reader.py",
    "src/dashboard/static/index.html": "src/dashboard/static/index.html",
    "deploy/Dockerfile": "Dockerfile",
    "deploy/requirements-dashboard.txt": "requirements.txt",
}

FORBIDDEN = ("config", "prompts", ".env", "cache", "agents", "brokers",
             "orchestrator", "options", "risk", "signals", "decisionlog")


def stage(out: Path) -> int:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for source, target in FILES.items():
        destination = out / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / source, destination)

    logs = REPO / "deploy" / "logs"
    if not logs.is_dir():
        print("run `python -m deploy.bundle_logs` first -- no sanitised logs")
        return 1
    shutil.copytree(logs, out / "logs")

    (out / ".dockerignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")

    # The staging directory is the last place to catch a mistake, because
    # after this it is an upload.
    staged = sorted(p for p in out.rglob("*") if p.is_file())
    bad = [p for p in staged
           if any(f in p.relative_to(out).as_posix() for f in FORBIDDEN)]
    for path in staged:
        print(f"  {path.relative_to(out).as_posix()}")
    if bad:
        print("\nREFUSING: forbidden paths staged:")
        for path in bad:
            print(f"  {path.relative_to(out).as_posix()}")
        shutil.rmtree(out)
        return 2

    print(f"\n{len(staged)} files staged in {out}")
    print("nothing from config/, prompts/, .env or the trading packages")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m deploy.stage")
    p.add_argument("--out", type=Path, required=True,
                   help="staging directory; MUST be outside the repository")
    args = p.parse_args(argv)
    if REPO in args.out.resolve().parents or args.out.resolve() == REPO:
        print("--out must be outside the repository; see this module's docstring")
        return 2
    return stage(args.out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
