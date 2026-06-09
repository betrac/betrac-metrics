#!/usr/bin/env python3
"""Map old dialog IDs (dialog_XXXX_YYYY) to new UUIDs from the BeTrac-Data-Upload repo.

Usage:
    # Build mapping CSV from data upload repo
    python scripts/map_ids.py build-mapping \
        --data-dir ../BeTrac-Data-Upload/data \
        --output id_mapping.csv

    # Convert a predictions JSONL from old IDs to new UUIDs
    python scripts/map_ids.py convert \
        --mapping id_mapping.csv \
        --input results/beyond-cascaded/summaries-beyond-qwen3-asr.jsonl \
        --output results/beyond-cascaded/summaries-beyond-qwen3-asr-mapped.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def build_mapping(data_dir: Path) -> dict[str, str]:
    """Scan dialog_*.json files and extract old_id -> new_uuid mapping."""
    mapping: dict[str, str] = {}
    for json_file in sorted(data_dir.rglob("dialog_*.json")):
        old_id = json_file.stem
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            new_id = data.get("id")
            if new_id:
                mapping[old_id] = str(new_id)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read {json_file}: {e}", file=sys.stderr)
    return mapping


def cmd_build_mapping(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory", file=sys.stderr)
        return 1

    mapping = build_mapping(data_dir)
    if not mapping:
        print("Error: no dialog_*.json files found", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["old_id", "new_id"])
        for old_id, new_id in sorted(mapping.items()):
            writer.writerow([old_id, new_id])

    print(f"Wrote {len(mapping)} mappings to {output}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    # Load mapping
    mapping: dict[str, str] = {}
    with open(args.mapping, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["old_id"]] = row["new_id"]
    print(f"Loaded {len(mapping)} ID mappings")

    # Convert JSONL
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            old_id = record.get("id", "")
            if old_id in mapping:
                record["id"] = mapping[old_id]
                record["original_id"] = old_id
                converted += 1
            else:
                skipped += 1
                if skipped <= 5:
                    print(f"Warning: no mapping for '{old_id}' (line {line_num})",
                          file=sys.stderr)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {converted} IDs, skipped {skipped}")
    print(f"Output: {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Map old dialog IDs to new UUIDs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-mapping
    bm = subparsers.add_parser("build-mapping", help="Build ID mapping CSV from dialog JSON files")
    bm.add_argument("--data-dir", required=True, help="Path to BeTrac-Data-Upload/data directory")
    bm.add_argument("--output", "-o", default="id_mapping.csv", help="Output CSV path")
    bm.set_defaults(func=cmd_build_mapping)

    # convert
    cv = subparsers.add_parser("convert", help="Convert JSONL from old IDs to new UUIDs")
    cv.add_argument("--mapping", required=True, help="ID mapping CSV (from build-mapping)")
    cv.add_argument("--input", required=True, help="Input JSONL with old IDs")
    cv.add_argument("--output", required=True, help="Output JSONL with new UUIDs")
    cv.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
