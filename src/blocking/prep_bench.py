"""Old vs new stage1.prepare(): identical outputs and time per source file.

The old code is read from a git ref (default c71a977, before the speed-up) into a temporary
package, so both versions run in the same process on the same data. For every source file of the
chosen countries it prints the row counts, column names, dtypes and a hash of every column for both
versions, whether they are equal, and the times (old, new, and a cached read when enabled).

Usage:
    python -m src.blocking.prep_bench --split test --countries India US France
"""
import argparse
import gc
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

from src.common.io import REPO_ROOT, scan_source
from . import stage1 as new
from .text import load_script_map

FILES = ("__init__.py", "text.py", "stage1.py", "tfidf_v0.py")


def load_old(ref: str):
    """Import src/blocking at `ref` as the package `prep_old`."""
    tmp = Path(tempfile.mkdtemp(prefix="prep_old_"))
    pkg = tmp / "prep_old"
    pkg.mkdir()
    for f in FILES:
        src = subprocess.run(["git", "show", f"{ref}:src/blocking/{f}"], cwd=REPO_ROOT,
                             check=True, capture_output=True, text=True).stdout
        (pkg / f).write_text(src)
    sys.path.insert(0, str(tmp))
    import prep_old.stage1 as old  # noqa: E402
    return old


def col_hash(s: pl.Series) -> str:
    """Order-sensitive hash of a column (row position included)."""
    h = s.to_frame().with_row_index().hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3)
    return f"{int(h.sum()) % (1 << 64):016x}"


def compare(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    ok = a.columns == b.columns and a.height == b.height
    print(f"    rows old {a.height:,} new {b.height:,} | columns equal: {a.columns == b.columns}")
    for c in a.columns:
        if c not in b.columns:
            print(f"    {c:12s} MISSING in new")
            ok = False
            continue
        same = a[c].dtype == b[c].dtype and bool(a[c].eq_missing(b[c]).all())
        ok &= same
        extra = f" (true: {a[c].sum():,} / {b[c].sum():,})" if a[c].dtype == pl.Boolean else ""
        print(f"    {c:12s} {str(a[c].dtype):8s} old {col_hash(a[c])} new {col_hash(b[c])} {'OK' if same else 'DIFF'}{extra}")
    return ok


def timed(f, *a):
    t = time.time()
    r = f(*a)
    return r, time.time() - t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--sources", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--ref", default="c71a977", help="git ref of the old code")
    ap.add_argument("--no-cache-check", action="store_true", help="skip the cached-read check")
    args = ap.parse_args()

    os.environ.pop("BLOCKING_PREP_CACHE", None)
    old = load_old(args.ref)
    tmap = load_script_map()
    cache_dir = Path(tempfile.mkdtemp(prefix="prep_cache_"))
    rows, all_ok = [], True
    for country in args.countries:
        for source in args.sources:
            scan_source(args.split, source, country=country).select(pl.len()).collect()  # warm the file cache
            print(f"[{args.split} S{source} {country}]", flush=True)
            b, t_new = timed(new.prepare, args.split, source, country, tmap)
            a, t_old = timed(old.prepare, args.split, source, country, tmap)
            ok = compare(a, b)
            del a
            t_cached = float("nan")
            if not args.no_cache_check:
                os.environ["BLOCKING_PREP_CACHE"] = str(cache_dir)
                new.prepare(args.split, source, country, tmap)  # writes the cache file
                c, t_cached = timed(new.prepare, args.split, source, country, tmap)
                os.environ.pop("BLOCKING_PREP_CACHE")
                same = c.equals(b)
                print(f"    cached read identical: {same}")
                ok &= same
                del c
            all_ok &= ok
            rows.append((country, source, b.height, t_old, t_new, t_cached, ok))
            print(f"    time old {t_old:.1f}s new {t_new:.1f}s cached read {t_cached:.1f}s -> {'IDENTICAL' if ok else 'DIFFERENT'}",
                  flush=True)
            del b
            gc.collect()
    print("\ncountry  src      rows    old_s   new_s  cached_s  identical")
    for c, s, n, to, tn, tc, ok in rows:
        print(f"{c:8s} S{s} {n:10,} {to:8.1f} {tn:7.1f} {tc:8.1f}  {ok}")
    for c in args.countries:
        r = [x for x in rows if x[0] == c]
        print(f"{c}: old {sum(x[3] for x in r):.0f}s -> new {sum(x[4] for x in r):.0f}s "
              f"-> cached {sum(x[5] for x in r):.0f}s")
    print("ALL IDENTICAL" if all_ok else "SOME OUTPUTS DIFFER", flush=True)


if __name__ == "__main__":
    main()
