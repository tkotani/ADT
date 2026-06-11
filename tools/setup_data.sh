#!/usr/bin/env bash
# ============================================================
# setup_data.sh — populate $ADT_ROOT/data/ via symlinks from $PTBANK
# ============================================================
# Usage:
#   PTBANK=$HOME/PTBANK ./setup_data.sh
#
# Reads MANIFEST (data_manifest.tsv) and creates symlinks:
#   $ADT_ROOT/<repo_path>  →  $PTBANK/<ptbank_name>
#
# If $PTBANK file is missing, prints download URL hint (Zenodo etc.).
# If $PTBANK file exists, symlink is created (replaces existing target).
# md5 column is verified when --verify is passed.
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADT_ROOT="${ADT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PTBANK="${PTBANK:-$ADT_ROOT/PTBANK}"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/data_manifest.tsv}"
VERIFY=0; [ "$1" = "--verify" ] && VERIFY=1

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest not found: $MANIFEST" >&2; exit 1
fi
mkdir -p "$PTBANK"
echo "ADT_ROOT = $ADT_ROOT"
echo "PTBANK   = $PTBANK"
echo ""

n_ok=0; n_missing=0; n_md5fail=0
while IFS=$'\t' read -r repo_path ptbank_name size_mb md5 note; do
  case "$repo_path" in \#*|"") continue;; esac
  src="$PTBANK/$ptbank_name"
  dst="$ADT_ROOT/$repo_path"
  mkdir -p "$(dirname "$dst")"
  if [ -f "$src" ]; then
    ln -sfn "$src" "$dst"
    if [ $VERIFY = 1 ] && [ -n "$md5" ] && [ "$md5" != "-" ]; then
      actual=$(md5sum "$src" | cut -d' ' -f1)
      if [ "$actual" = "$md5" ]; then
        echo "[OK]      $repo_path ($size_mb MB, md5 ok)"
      else
        echo "[MD5 FAIL] $repo_path (expected $md5, got $actual)"
        n_md5fail=$((n_md5fail+1))
      fi
    else
      echo "[OK]      $repo_path -> $ptbank_name"
    fi
    n_ok=$((n_ok+1))
  else
    echo "[MISSING] $repo_path  (need: $PTBANK/$ptbank_name, $size_mb MB) — $note"
    n_missing=$((n_missing+1))
  fi
done < "$MANIFEST"

echo ""
echo "Summary: $n_ok linked, $n_missing missing, $n_md5fail md5-fail"
if [ $n_missing -gt 0 ]; then
  echo ""
  echo "To populate $PTBANK:"
  echo "  Option 1 (Zenodo, future):"
  echo "    DOI=10.5281/zenodo.XXXXXXX (TBD when uploaded)"
  echo "    wget https://zenodo.org/record/XXXXXXX/files/adt_paper_data.tar.gz"
  echo "    tar -xzf adt_paper_data.tar.gz -C $PTBANK/"
  echo "  Option 2 (local copy from kr1/kr2):"
  echo "    rsync -av takao@kr2:~/ADT/Drugs/data/freeorder_v26/ $PTBANK/   # (with renaming)"
  echo "  Then re-run: $0"
fi
