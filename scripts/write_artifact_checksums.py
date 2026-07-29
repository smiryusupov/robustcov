#!/usr/bin/env python3
"""Write deterministic SHA-256 checksums for release artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dist/SHA256SUMS"))
    args = parser.parse_args()

    files = sorted({path.resolve() for path in args.artifacts}, key=lambda path: path.name)
    missing = [path for path in files if not path.is_file()]
    if missing:
        for path in missing:
            print(f"missing artifact: {path}", file=sys.stderr)
        return 2
    duplicate_names = {path.name for path in files if sum(p.name == path.name for p in files) > 1}
    if duplicate_names:
        print(f"duplicate artifact filenames: {sorted(duplicate_names)}", file=sys.stderr)
        return 2

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{_digest(path)}  {path.name}" for path in files]
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
