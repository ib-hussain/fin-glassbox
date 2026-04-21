from __future__ import annotations

import argparse
import csv
import re
import heapq
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import orjson  # type: ignore
    HAS_ORJSON = True
except ImportError:
    orjson = None
    HAS_ORJSON = False


# ============================================================
# PATHS
# ============================================================

dataPath = Path(os.getenv("dataPathGlobal", "data"))

RAW_INPUT_DIR = dataPath / "sec_edgar" / "raw" / "bulk" / "companyfacts_extracted"
OUT_ROOT = dataPath / "sec_edgar" / "processed" / "companyfacts"

PARTS_DIR = OUT_ROOT / "companyfacts_flat"
SHARD_DIR = OUT_ROOT / "_shards"
TMP_DIR = OUT_ROOT / "_tmp"

INVENTORY_FINAL = OUT_ROOT / "companyfacts_inventory.csv"
ERRORS_FINAL = OUT_ROOT / "companyfacts_errors.csv"
SUMMARY_FINAL = OUT_ROOT / "companyfacts_summary.json"


# ============================================================
# CSV HEADERS
# ============================================================

FACTS_HEADER = [
    "source_json",
    "cik",
    "entity_name",
    "taxonomy",
    "concept",
    "label",
    "description",
    "unit",
    "fy",
    "fp",
    "form",
    "filed",
    "end",
    "frame",
    "accn",
    "val",
]

INVENTORY_HEADER = [
    "source_json",
    "file_size_bytes",
    "cik",
    "entity_name",
    "n_taxonomies",
    "n_concepts",
    "n_units",
    "n_observations",
    "earliest_filed",
    "latest_filed",
    "earliest_end",
    "latest_end",
    "status",
    "error_message",
]

ERRORS_HEADER = [
    "source_json",
    "error_type",
    "error_message",
]


# ============================================================
# HELPERS
# ============================================================
PART_ID_RE = re.compile(r"(?:facts|inventory|errors)_part_(\d{5})\.csv$", re.IGNORECASE)


def load_existing_processed_keys(inventory_csv: Path) -> tuple[set[str], set[str]]:
    """
    Returns:
      processed_relpaths: source_json values from existing inventory
      processed_basenames: file basenames from source_json

    We keep both because if the user copied files back with slightly different
    folder nesting, basename-based skipping still works for SEC companyfacts,
    whose filenames are effectively unique per CIK.
    """
    processed_relpaths: set[str] = set()
    processed_basenames: set[str] = set()

    if not inventory_csv.exists():
        return processed_relpaths, processed_basenames

    with inventory_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "source_json" not in reader.fieldnames:
            raise RuntimeError(
                f"Existing inventory file does not have expected 'source_json' column: {inventory_csv}"
            )

        for row in reader:
            src = (row.get("source_json") or "").strip()
            if not src:
                continue
            src = src.replace("\\", "/")
            processed_relpaths.add(src)
            processed_basenames.add(Path(src).name)

    return processed_relpaths, processed_basenames


def next_available_part_id(parts_dir: Path) -> int:
    """
    Finds the next shard/part number by scanning existing part files.
    """
    max_id = 0
    if parts_dir.exists():
        for p in parts_dir.glob("*_part_*.csv"):
            m = PART_ID_RE.match(p.name)
            if m:
                max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def load_previous_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def human_bytes(n: int | float) -> str:
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if n < 1024.0 or unit == units[-1]:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} B"


def now_ts() -> float:
    return time.time()


def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


def json_loads_bytes(raw: bytes) -> Any:
    if HAS_ORJSON:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def ensure_clean_output_root(overwrite: bool) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for p in [PARTS_DIR, SHARD_DIR, TMP_DIR, INVENTORY_FINAL, ERRORS_FINAL, SUMMARY_FINAL]:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def discover_json_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found under: {input_dir}")
    return files


def greedy_partition(files: list[Path], shard_count: int) -> list[list[Path]]:
    """
    Greedy size-balanced partitioning.
    Sort biggest files first, always assign next file to currently lightest shard.
    """
    shard_count = max(1, min(shard_count, len(files)))
    shards: list[list[Path]] = [[] for _ in range(shard_count)]
    heap: list[tuple[int, int]] = [(0, i) for i in range(shard_count)]
    heapq.heapify(heap)

    file_sizes = []
    for p in files:
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        file_sizes.append((size, p))

    file_sizes.sort(key=lambda x: x[0], reverse=True)

    for size, path in file_sizes:
        total_size, shard_idx = heapq.heappop(heap)
        shards[shard_idx].append(path)
        heapq.heappush(heap, (total_size + size, shard_idx))

    return [s for s in shards if s]


def write_csv_header(path: Path, header: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def copy_fileobj_in_chunks(src_fh, dst_fh, chunk_size: int = 16 * 1024 * 1024) -> int:
    copied = 0
    while True:
        chunk = src_fh.read(chunk_size)
        if not chunk:
            break
        dst_fh.write(chunk)
        copied += len(chunk)
    return copied


def merge_csv_parts(
    part_paths: list[Path],
    out_path: Path,
    progress_prefix: str,
    progress_every_mb: int = 512,
) -> None:
    if not part_paths:
        raise ValueError(f"No part files provided for merge into {out_path}")

    total_bytes = sum(p.stat().st_size for p in part_paths if p.exists())
    done_bytes = 0
    next_report = progress_every_mb * 1024 * 1024
    start = now_ts()

    tmp_out = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_out.exists():
        tmp_out.unlink()

    with tmp_out.open("wb") as out_fh:
        for idx, part in enumerate(part_paths):
            if not part.exists():
                continue

            with part.open("rb") as in_fh:
                if idx > 0:
                    # Skip header line for all but first file
                    in_fh.readline()
                copied = copy_fileobj_in_chunks(in_fh, out_fh)
                done_bytes += copied

                if done_bytes >= next_report or done_bytes == total_bytes:
                    elapsed = max(now_ts() - start, 1e-9)
                    speed = done_bytes / elapsed
                    pct = (done_bytes / total_bytes * 100.0) if total_bytes else 100.0
                    eta = (total_bytes - done_bytes) / speed if speed > 0 else 0.0
                    print(
                        f"{progress_prefix} | {pct:6.2f}% | "
                        f"{human_bytes(done_bytes)} / {human_bytes(total_bytes)} | "
                        f"{human_bytes(speed)}/s | ETA {fmt_elapsed(eta)}",
                        flush=True,
                    )
                    next_report += progress_every_mb * 1024 * 1024

    tmp_out.replace(out_path)


def safe_min_date(current: str, new_val: str) -> str:
    if not new_val:
        return current
    if not current:
        return new_val
    return min(current, new_val)


def safe_max_date(current: str, new_val: str) -> str:
    if not new_val:
        return current
    if not current:
        return new_val
    return max(current, new_val)


# ============================================================
# WORKER
# ============================================================

def process_shard(
    shard_id: int,
    files: list[str],
    input_root_str: str,
    shard_dir_str: str,
) -> dict[str, Any]:
    """
    Each worker processes a list of JSON files and writes:
      - one facts CSV part
      - one inventory CSV shard
      - one errors CSV shard
    """
    input_root = Path(input_root_str)
    shard_dir = Path(shard_dir_str)

    facts_part_path = shard_dir / f"facts_part_{shard_id:05d}.csv"
    inventory_part_path = shard_dir / f"inventory_part_{shard_id:05d}.csv"
    errors_part_path = shard_dir / f"errors_part_{shard_id:05d}.csv"

    write_csv_header(facts_part_path, FACTS_HEADER)
    write_csv_header(inventory_part_path, INVENTORY_HEADER)
    write_csv_header(errors_part_path, ERRORS_HEADER)

    facts_rows = 0
    inventory_rows = 0
    error_rows = 0
    ok_files = 0
    failed_files = 0

    with (
        facts_part_path.open("a", newline="", encoding="utf-8") as facts_fh,
        inventory_part_path.open("a", newline="", encoding="utf-8") as inv_fh,
        errors_part_path.open("a", newline="", encoding="utf-8") as err_fh,
    ):
        facts_writer = csv.writer(facts_fh)
        inv_writer = csv.writer(inv_fh)
        err_writer = csv.writer(err_fh)

        for file_str in files:
            path = Path(file_str)

            try:
                rel_path = str(path.relative_to(input_root)).replace("\\", "/")
            except Exception:
                rel_path = str(path).replace("\\", "/")

            file_size = 0
            try:
                file_size = path.stat().st_size
            except Exception:
                pass

            try:
                raw = path.read_bytes()
                obj = json_loads_bytes(raw)

                cik = obj.get("cik", "")
                entity_name = obj.get("entityName", "")
                facts = obj.get("facts", {}) or {}

                n_taxonomies = 0
                n_concepts = 0
                n_units = 0
                n_observations = 0
                earliest_filed = ""
                latest_filed = ""
                earliest_end = ""
                latest_end = ""

                if not isinstance(facts, dict) or not facts:
                    inv_writer.writerow([
                        rel_path,
                        file_size,
                        cik,
                        entity_name,
                        0,
                        0,
                        0,
                        0,
                        "",
                        "",
                        "",
                        "",
                        "no_facts",
                        "",
                    ])
                    inventory_rows += 1
                    ok_files += 1
                    continue

                for taxonomy, concept_map in facts.items():
                    if not isinstance(concept_map, dict):
                        continue

                    n_taxonomies += 1

                    for concept, concept_payload in concept_map.items():
                        if not isinstance(concept_payload, dict):
                            continue

                        n_concepts += 1

                        label = concept_payload.get("label", "")
                        description = concept_payload.get("description", "")
                        units = concept_payload.get("units", {}) or {}

                        if not isinstance(units, dict):
                            continue

                        for unit, obs_list in units.items():
                            n_units += 1

                            if not isinstance(obs_list, list):
                                continue

                            for obs in obs_list:
                                if not isinstance(obs, dict):
                                    continue

                                fy = obs.get("fy", "")
                                fp = obs.get("fp", "")
                                form = obs.get("form", "")
                                filed = obs.get("filed", "")
                                end = obs.get("end", "")
                                frame = obs.get("frame", "")
                                accn = obs.get("accn", "")
                                val = obs.get("val", "")

                                earliest_filed = safe_min_date(earliest_filed, filed)
                                latest_filed = safe_max_date(latest_filed, filed)
                                earliest_end = safe_min_date(earliest_end, end)
                                latest_end = safe_max_date(latest_end, end)

                                facts_writer.writerow([
                                    rel_path,
                                    cik,
                                    entity_name,
                                    taxonomy,
                                    concept,
                                    label,
                                    description,
                                    unit,
                                    fy,
                                    fp,
                                    form,
                                    filed,
                                    end,
                                    frame,
                                    accn,
                                    val,
                                ])
                                facts_rows += 1
                                n_observations += 1

                inv_writer.writerow([
                    rel_path,
                    file_size,
                    cik,
                    entity_name,
                    n_taxonomies,
                    n_concepts,
                    n_units,
                    n_observations,
                    earliest_filed,
                    latest_filed,
                    earliest_end,
                    latest_end,
                    "ok",
                    "",
                ])
                inventory_rows += 1
                ok_files += 1

            except Exception as exc:
                failed_files += 1
                error_rows += 1

                err_writer.writerow([
                    rel_path,
                    type(exc).__name__,
                    str(exc),
                ])

                inv_writer.writerow([
                    rel_path,
                    file_size,
                    "",
                    "",
                    0,
                    0,
                    0,
                    0,
                    "",
                    "",
                    "",
                    "",
                    "error",
                    str(exc),
                ])
                inventory_rows += 1

    return {
        "shard_id": shard_id,
        "facts_part": str(facts_part_path),
        "inventory_part": str(inventory_part_path),
        "errors_part": str(errors_part_path),
        "input_files": len(files),
        "ok_files": ok_files,
        "failed_files": failed_files,
        "facts_rows": facts_rows,
        "inventory_rows": inventory_rows,
        "error_rows": error_rows,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory + flatten SEC companyfacts JSON files into partitioned CSVs."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Incremental mode: skip files already recorded in existing companyfacts_inventory.csv and append only new files.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(RAW_INPUT_DIR),
        help="Directory containing extracted SEC companyfacts JSON files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes. 0 = auto.",
    )
    parser.add_argument(
        "--shard-multiplier",
        type=int,
        default=4,
        help="Number of shards ~= workers * shard_multiplier. More shards improves load balance.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous processed outputs before running.",
    )
    return parser.parse_args()


def choose_worker_count(requested: int) -> int:
    if requested and requested > 0:
        return requested
    cpu = os.cpu_count() or 4
    if cpu <= 2:
        return 1
    return max(1, cpu - 1)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    workers = choose_worker_count(args.workers)

    ensure_clean_output_root(args.overwrite)

    print(f"Input directory : {input_dir}", flush=True)
    print(f"Output root     : {OUT_ROOT}", flush=True)
    print(f"JSON parser     : {'orjson' if HAS_ORJSON else 'stdlib json'}", flush=True)
    print(f"Workers         : {workers}", flush=True)
    print(f"Resume mode     : {args.resume}", flush=True)

    overall_start = now_ts()

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------
    discover_start = now_ts()
    all_files = discover_json_files(input_dir)
    all_input_bytes = sum(p.stat().st_size for p in all_files)
    discover_elapsed = now_ts() - discover_start

    print(
        f"Discovered {len(all_files):,} companyfacts JSON files "
        f"({human_bytes(all_input_bytes)}) in {fmt_elapsed(discover_elapsed)}",
        flush=True,
    )

    # --------------------------------------------------------
    # Incremental resume filtering
    # --------------------------------------------------------
    previous_summary = load_previous_summary(SUMMARY_FINAL)

    processed_relpaths: set[str] = set()
    processed_basenames: set[str] = set()
    files = all_files
    skipped_existing = 0

    if args.resume and INVENTORY_FINAL.exists():
        processed_relpaths, processed_basenames = load_existing_processed_keys(INVENTORY_FINAL)

        filtered: list[Path] = []
        for p in all_files:
            rel = str(p.relative_to(input_dir)).replace("\\", "/")
            base = p.name

            if rel in processed_relpaths or base in processed_basenames:
                skipped_existing += 1
                continue

            filtered.append(p)

        files = filtered

        print(
            f"Existing processed files detected: {len(processed_basenames):,} "
            f"(skipping {skipped_existing:,} already processed files)",
            flush=True,
        )

    if not files:
        print("No new companyfacts JSON files to process. Nothing to do.", flush=True)
        return

    new_input_bytes = sum(p.stat().st_size for p in files)
    print(
        f"New files to process: {len(files):,} "
        f"({human_bytes(new_input_bytes)})",
        flush=True,
    )

    # --------------------------------------------------------
    # Partition into shards
    # --------------------------------------------------------
    shard_count = max(1, min(len(files), workers * max(1, args.shard_multiplier)))
    shards = greedy_partition(files, shard_count)

    first_part_id = next_available_part_id(PARTS_DIR)

    print(f"Shard count       : {len(shards)}", flush=True)
    print(f"Starting part id  : {first_part_id:05d}", flush=True)
    print(f"Parts directory   : {PARTS_DIR}", flush=True)

    # --------------------------------------------------------
    # Process shards in parallel
    # --------------------------------------------------------
    futures = []
    shard_start = now_ts()

    total_files_done = 0
    total_ok_files = 0
    total_failed_files = 0
    total_fact_rows = 0
    total_inventory_rows = 0
    total_error_rows = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for offset, shard_files in enumerate(shards):
            shard_id = first_part_id + offset
            futures.append(
                executor.submit(
                    process_shard,
                    shard_id,
                    [str(p) for p in shard_files],
                    str(input_dir),
                    str(PARTS_DIR),
                )
            )

        completed = 0
        total_shards = len(futures)

        for fut in as_completed(futures):
            result = fut.result()
            completed += 1

            total_files_done += result["input_files"]
            total_ok_files += result["ok_files"]
            total_failed_files += result["failed_files"]
            total_fact_rows += result["facts_rows"]
            total_inventory_rows += result["inventory_rows"]
            total_error_rows += result["error_rows"]

            elapsed = max(now_ts() - shard_start, 1e-9)
            files_per_sec = total_files_done / elapsed
            rows_per_sec = total_fact_rows / elapsed if total_fact_rows else 0.0

            print(
                f"[process] shards {completed}/{total_shards} | "
                f"new files {total_files_done:,}/{len(files):,} | "
                f"ok {total_ok_files:,} | failed {total_failed_files:,} | "
                f"facts rows {total_fact_rows:,} | "
                f"{files_per_sec:,.1f} files/s | {rows_per_sec:,.1f} rows/s",
                flush=True,
            )

    process_elapsed = now_ts() - shard_start

    # --------------------------------------------------------
    # Rebuild final inventory and errors from all shard parts
    # --------------------------------------------------------
    inventory_parts = sorted(PARTS_DIR.glob("inventory_part_*.csv"))
    error_parts = sorted(PARTS_DIR.glob("errors_part_*.csv"))

    print("Rebuilding cumulative inventory CSV...", flush=True)
    merge_csv_parts(inventory_parts, INVENTORY_FINAL, "[merge inventory]")

    print("Rebuilding cumulative errors CSV...", flush=True)
    merge_csv_parts(error_parts, ERRORS_FINAL, "[merge errors]")

    # --------------------------------------------------------
    # Facts parts remain partitioned output
    # --------------------------------------------------------
    facts_parts = sorted(PARTS_DIR.glob("facts_part_*.csv"))

    previous_total_fact_rows = int(previous_summary.get("total_fact_rows", 0) or 0)
    previous_total_ok_files = int(previous_summary.get("total_ok_files", 0) or 0)
    previous_total_failed_files = int(previous_summary.get("total_failed_files", 0) or 0)

    cumulative_total_fact_rows = previous_total_fact_rows + total_fact_rows
    cumulative_total_ok_files = previous_total_ok_files + total_ok_files
    cumulative_total_failed_files = previous_total_failed_files + total_failed_files

    summary = {
        "input_dir": str(input_dir),
        "output_root": str(OUT_ROOT),
        "used_orjson": HAS_ORJSON,
        "workers": workers,
        "resume_mode": args.resume,
        "shard_count_this_run": len(shards),
        "starting_part_id_this_run": first_part_id,
        "json_files_discovered_total": len(all_files),
        "json_files_processed_this_run": len(files),
        "json_files_skipped_as_existing": skipped_existing,
        "total_input_bytes_all_visible": all_input_bytes,
        "total_input_bytes_processed_this_run": new_input_bytes,
        "total_ok_files_this_run": total_ok_files,
        "total_failed_files_this_run": total_failed_files,
        "total_inventory_rows_this_run": total_inventory_rows,
        "total_error_rows_this_run": total_error_rows,
        "total_fact_rows_this_run": total_fact_rows,
        "total_ok_files": cumulative_total_ok_files,
        "total_failed_files": cumulative_total_failed_files,
        "total_fact_rows": cumulative_total_fact_rows,
        "facts_parts": [str(p) for p in facts_parts],
        "inventory_csv": str(INVENTORY_FINAL),
        "errors_csv": str(ERRORS_FINAL),
        "timing": {
            "processing_seconds_this_run": process_elapsed,
            "total_seconds_this_run": now_ts() - overall_start,
        },
    }

    with SUMMARY_FINAL.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_elapsed = now_ts() - overall_start

    print("\n=== COMPANYFACTS PIPELINE COMPLETE ===", flush=True)
    print(f"Inventory CSV   : {INVENTORY_FINAL}", flush=True)
    print(f"Errors CSV      : {ERRORS_FINAL}", flush=True)
    print(f"Facts parts     : {len(facts_parts)} file(s) in {PARTS_DIR}", flush=True)
    print(f"Summary JSON    : {SUMMARY_FINAL}", flush=True)
    print(f"All visible JSON : {len(all_files):,}", flush=True)
    print(f"Processed this run: {len(files):,}", flush=True)
    print(f"Skipped existing : {skipped_existing:,}", flush=True)
    print(f"New fact rows    : {total_fact_rows:,}", flush=True)
    print(f"Cumulative rows  : {cumulative_total_fact_rows:,}", flush=True)
    print(f"Elapsed          : {fmt_elapsed(total_elapsed)}", flush=True)

if __name__ == "__main__":
    # Required for safe multiprocessing on Windows
    main()