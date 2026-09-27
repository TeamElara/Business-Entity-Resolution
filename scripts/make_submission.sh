#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 && ( $1 == --help || $1 == -h ) ]]; then
  echo "Usage: scripts/make_submission.sh <output_dir>"
  echo "<output_dir> must contain matching_results.tsv and candidate_pairs.tsv."
  exit 0
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: scripts/make_submission.sh <output_dir>" >&2
  echo "<output_dir> must contain matching_results.tsv and candidate_pairs.tsv." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$(cd "$1" && pwd)"

if [[ -n ${PYTHON:-} ]]; then
  python_bin="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  python_bin=python3
elif command -v python >/dev/null 2>&1; then
  python_bin=python
else
  echo "Python 3.11+ is required (or set PYTHON to its executable)." >&2
  exit 1
fi

"$python_bin" - "$repo_root" "$output_dir" <<'PY'
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
archive = output / "TeamElara_submission.zip"
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required to package this submission")
required = {
    "matching_results.tsv": b"source1_entity_id\tmatched_entity_ids",
    "candidate_pairs.tsv": b"source1_entity_id\tcandidate_entity_ids",
}

if archive.exists():
    raise SystemExit(f"Archive already exists; move it before rebuilding: {archive}")
for name, header in required.items():
    path = output / name
    if not path.is_file() or path.stat().st_size <= len(header):
        raise SystemExit(f"Missing or empty output: {path}")
    with path.open("rb") as stream:
        if stream.readline().rstrip(b"\r\n") != header:
            raise SystemExit(f"Unexpected TSV header: {path}")

tracked = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"])
paths = [Path(p.decode("utf-8")) for p in tracked.split(b"\0") if p]
code_roots = ("src/", "utils/", "docs/", "tests/", "scripts/")
code_files = [
    p for p in paths
    if p.as_posix().startswith(code_roots)
    or p.as_posix() in {"README.md", "RUN.md", "requirements.txt"}
]
for name in ("README.md", "RUN.md", "requirements.txt", "src/__init__.py", "utils/validate_submission.py"):
    if Path(name) not in code_files:
        raise SystemExit(f"Required code file is not tracked: {name}")
if Path("Documentation_template.md") not in paths:
    raise SystemExit("Required methodology is not tracked: Documentation_template.md")

tmp_name = None
try:
    with tempfile.NamedTemporaryFile(prefix=".submission-", suffix=".zip", dir=output, delete=False) as temp:
        tmp_name = temp.name
    with zipfile.ZipFile(tmp_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for name in required:
            zf.write(output / name, f"output/{name}")
        for path in sorted(code_files):
            zf.write(repo / path, f"code/business_entity_resolution/{path.as_posix()}")
        zf.write(repo / "Documentation_template.md", "Documentation_template.md")
        zf.write(repo / "RUN.md", "RUN.md")
    with zipfile.ZipFile(tmp_name) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f"Corrupt zip member: {bad}")
        names = set(zf.namelist())
        for name in required:
            if f"output/{name}" not in names:
                raise SystemExit(f"Missing zip member: output/{name}")
    os.replace(tmp_name, archive)
    tmp_name = None
finally:
    if tmp_name and os.path.exists(tmp_name):
        os.unlink(tmp_name)

print(archive)
print("TeamElara_submission.zip")
print("|-- output/")
for name in required:
    print(f"|   |-- {name}")
print("|-- code/business_entity_resolution/")
for name in ("src/", "utils/", "docs/", "tests/", "scripts/", "README.md", "RUN.md", "requirements.txt"):
    print(f"|   |-- {name}")
print("|-- Documentation_template.md")
print("`-- RUN.md")
print(f"Size: {archive.stat().st_size:,} bytes ({archive.stat().st_size / (1024 * 1024):.2f} MiB)")
print("TSV headers checked; run validate_submission.py --check-ids on the final files.")
PY
