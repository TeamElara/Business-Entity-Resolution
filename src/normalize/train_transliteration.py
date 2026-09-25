"""Train a local Hindi-to-Latin token map using non-validation true pairs.

Reads raw training TSVs directly; Phase 8 Parquet files are not required.
The generated map is under the gitignored data directory and is not committed
to the team repository.
"""

import argparse
from pathlib import Path

from .transliteration import (
    DEFAULT_MAP_PATH,
    train_and_write_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_MAP_PATH)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    args = parser.parse_args()
    if not 0 < args.min_confidence <= 1:
        parser.error("--min-confidence must be in (0, 1]")
    stats = train_and_write_map(args.out, args.min_confidence)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
