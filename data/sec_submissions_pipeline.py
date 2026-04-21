from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import re
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

RAW_INPUT_DIR = dataPath / "sec_edgar" / "raw" / "bulk" / "submissions_extracted"
OUT_ROOT = dataPath / "sec_edgar" / "processed" / "submissions"

PARTS_DIR = OUT_ROOT / "submissions_flat"
TMP_DIR = OUT_ROOT / "_tmp"

INVENTORY_FINAL = OUT_ROOT / "submissions_inventory.csv"
ERRORS_FINAL = OUT_ROOT / "submissions_errors.csv"
SUMMARY_FINAL = OUT_ROOT / "submissions_summary.json"


# ============================================================
# CSV HEADERS
# ============================================================

INVENTORY_HEADER = [
    "source_json",
    "file_size_bytes",
    "cik",
    "entity_name",
    "entity_type",
    "n_tickers",
    "n_exchanges",
    "n_former_names",
    "n_recent_filings",
    "n_file_refs",
    "earliest_filing_date",
    "latest_filing_date",
    "status",
    "error_message",
]

ERRORS_HEADER = [
    "source_json",
    "error_type",
    "error_message",
]

ENTITIES_HEADER = [
    "source_json",
    "cik",
    "entityType",
    "sic",
    "sicDescription",
    "ownerOrg",
    "insiderTransactionForOwnerExists",
    "insiderTransactionForIssuerExists",
    "name",
    "tickers",
    "exchanges",
    "ein",
    "description",
    "website",
    "investorWebsite",
    "category",
    "fiscalYearEnd",
    "stateOfIncorporation",
    "stateOfIncorporationDescription",
    "phone",
    "flags",
    "mailing_street1",
    "mailing_street2",
    "mailing_city",
    "mailing_stateOrCountry",
    "mailing_stateOrCountryDescription",
    "mailing_zipCode",
    "business_street1",
    "business_street2",
    "business_city",
    "business_stateOrCountry",
    "business_stateOrCountryDescription",
    "business_zipCode",
]

RECENT_FILINGS_HEADER = [
    "source_json",
    "cik",
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "act",
    "form",
    "fileNumber",
    "filmNumber",
    "items",
    "size",
    "isXBRL",
    "isInlineXBRL",
    "primaryDocument",
    "primaryDocDescription",
]

FILING_FILES_HEADER = [
    "source_json",
    "cik",
    "name",
    "filingCount",
    "filingFrom",
    "filingTo",
]

FORMER_NAMES_HEADER = [
    "source_json",
    "cik",
    "name",
    "from_date",
    "to_date",
]


# ============================================================
# HELPERS
# ============================================================

PART_ID_RE = re.compile(
    r"(?:inventory|errors|entities|recent_filings|filing_files|former_names)_part_(\d{5})\.csv$",
    re.IGNORECASE,
)


class LowDiskSpaceError(RuntimeError):
    pass


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
                    in_fh.readline()  # skip header in later parts
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


def list_to_pipe_str(value: Any) -> str:
    if isinstance(value, list):
        return "|".join("" if v is None else str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def boolish_to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def get_address_field(address_obj: dict[str, Any] | None, key: str) -> str:
    if not isinstance(address_obj, dict):
        return ""
    v = address_obj.get(key, "")
    return "" if v is None else str(v)


def choose_worker_count(requested: int) -> int:
    if requested and requested > 0:
        return requested
    cpu = os.cpu_count() or 4
    if cpu <= 2:
        return 1
    return max(1, cpu - 1)


def ensure_dirs(overwrite: bool) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for p in [PARTS_DIR, TMP_DIR, INVENTORY_FINAL, ERRORS_FINAL, SUMMARY_FINAL]:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def discover_json_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found under: {input_dir}")
    return files


def greedy_partition(files: list[Path], shard_count: int) -> list[list[Path]]:
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


def next_available_part_id(parts_dir: Path) -> int:
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


def iter_existing_inventory_sources() -> list[Path]:
    paths: list[Path] = []
    if INVENTORY_FINAL.exists():
        paths.append(INVENTORY_FINAL)
    paths.extend(sorted(PARTS_DIR.glob("inventory_part_*.csv")))
    return paths


def load_existing_processed_keys() -> tuple[set[str], set[str]]:
    """
    Returns:
      processed_relpaths
      processed_basenames

    We load from both the final merged inventory and any shard inventory parts.
    This makes resume robust even if the previous run terminated before final merge.
    """
    processed_relpaths: set[str] = set()
    processed_basenames: set[str] = set()

    for inventory_csv in iter_existing_inventory_sources():
        try:
            with inventory_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "source_json" not in reader.fieldnames:
                    continue

                for row in reader:
                    src = (row.get("source_json") or "").strip()
                    if not src:
                        continue
                    src = src.replace("\\", "/")
                    processed_relpaths.add(src)
                    processed_basenames.add(Path(src).name)
        except Exception:
            # corrupted partial inventory shards should not kill resume
            continue

    return processed_relpaths, processed_basenames


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def check_disk_or_raise(base_path: Path, min_free_bytes: int, context: str) -> None:
    available = free_bytes(base_path)
    if available < min_free_bytes:
        raise LowDiskSpaceError(
            f"Low disk space during {context}. "
            f"Available={human_bytes(available)}, required minimum={human_bytes(min_free_bytes)}"
        )


def safe_list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def safe_list_item(lst: Any, idx: int) -> str:
    if not isinstance(lst, list):
        return ""
    if idx >= len(lst):
        return ""
    val = lst[idx]
    return "" if val is None else str(val)


# ============================================================
# WORKER
# ============================================================

def process_shard(
    shard_id: int,
    files: list[str],
    input_root_str: str,
    parts_dir_str: str,
    min_free_bytes: int,
    disk_check_every_files: int,
) -> dict[str, Any]:
    input_root = Path(input_root_str)
    parts_dir = Path(parts_dir_str)

    inventory_part_path = parts_dir / f"inventory_part_{shard_id:05d}.csv"
    errors_part_path = parts_dir / f"errors_part_{shard_id:05d}.csv"
    entities_part_path = parts_dir / f"entities_part_{shard_id:05d}.csv"
    recent_filings_part_path = parts_dir / f"recent_filings_part_{shard_id:05d}.csv"
    filing_files_part_path = parts_dir / f"filing_files_part_{shard_id:05d}.csv"
    former_names_part_path = parts_dir / f"former_names_part_{shard_id:05d}.csv"

    write_csv_header(inventory_part_path, INVENTORY_HEADER)
    write_csv_header(errors_part_path, ERRORS_HEADER)
    write_csv_header(entities_part_path, ENTITIES_HEADER)
    write_csv_header(recent_filings_part_path, RECENT_FILINGS_HEADER)
    write_csv_header(filing_files_part_path, FILING_FILES_HEADER)
    write_csv_header(former_names_part_path, FORMER_NAMES_HEADER)

    inventory_rows = 0
    error_rows = 0
    entity_rows = 0
    recent_rows = 0
    filing_file_rows = 0
    former_name_rows = 0
    ok_files = 0
    failed_files = 0

    with (
        inventory_part_path.open("a", newline="", encoding="utf-8") as inv_fh,
        errors_part_path.open("a", newline="", encoding="utf-8") as err_fh,
        entities_part_path.open("a", newline="", encoding="utf-8") as ent_fh,
        recent_filings_part_path.open("a", newline="", encoding="utf-8") as recent_fh,
        filing_files_part_path.open("a", newline="", encoding="utf-8") as file_fh,
        former_names_part_path.open("a", newline="", encoding="utf-8") as former_fh,
    ):
        inv_writer = csv.writer(inv_fh)
        err_writer = csv.writer(err_fh)
        ent_writer = csv.writer(ent_fh)
        recent_writer = csv.writer(recent_fh)
        file_writer = csv.writer(file_fh)
        former_writer = csv.writer(former_fh)

        for file_idx, file_str in enumerate(files, start=1):
            if file_idx % max(1, disk_check_every_files) == 0:
                check_disk_or_raise(parts_dir, min_free_bytes, f"processing shard {shard_id}")

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

                cik = "" if obj.get("cik") is None else str(obj.get("cik"))
                entity_name = "" if obj.get("name") is None else str(obj.get("name"))
                entity_type = "" if obj.get("entityType") is None else str(obj.get("entityType"))

                tickers = obj.get("tickers", [])
                exchanges = obj.get("exchanges", [])
                former_names = obj.get("formerNames", [])
                filings = obj.get("filings", {}) or {}
                recent = filings.get("recent", {}) or {}
                filing_files = filings.get("files", []) or []

                n_tickers = safe_list_len(tickers)
                n_exchanges = safe_list_len(exchanges)
                n_former_names = safe_list_len(former_names)
                n_file_refs = safe_list_len(filing_files)

                # ------------------------------------------------
                # entity metadata row
                # ------------------------------------------------
                mailing = obj.get("addresses", {}).get("mailing", {}) if isinstance(obj.get("addresses"), dict) else {}
                business = obj.get("addresses", {}).get("business", {}) if isinstance(obj.get("addresses"), dict) else {}

                ent_writer.writerow([
                    rel_path,
                    cik,
                    entity_type,
                    "" if obj.get("sic") is None else str(obj.get("sic")),
                    "" if obj.get("sicDescription") is None else str(obj.get("sicDescription")),
                    "" if obj.get("ownerOrg") is None else str(obj.get("ownerOrg")),
                    boolish_to_str(obj.get("insiderTransactionForOwnerExists")),
                    boolish_to_str(obj.get("insiderTransactionForIssuerExists")),
                    entity_name,
                    list_to_pipe_str(tickers),
                    list_to_pipe_str(exchanges),
                    "" if obj.get("ein") is None else str(obj.get("ein")),
                    "" if obj.get("description") is None else str(obj.get("description")),
                    "" if obj.get("website") is None else str(obj.get("website")),
                    "" if obj.get("investorWebsite") is None else str(obj.get("investorWebsite")),
                    "" if obj.get("category") is None else str(obj.get("category")),
                    "" if obj.get("fiscalYearEnd") is None else str(obj.get("fiscalYearEnd")),
                    "" if obj.get("stateOfIncorporation") is None else str(obj.get("stateOfIncorporation")),
                    "" if obj.get("stateOfIncorporationDescription") is None else str(obj.get("stateOfIncorporationDescription")),
                    "" if obj.get("phone") is None else str(obj.get("phone")),
                    "" if obj.get("flags") is None else str(obj.get("flags")),
                    get_address_field(mailing, "street1"),
                    get_address_field(mailing, "street2"),
                    get_address_field(mailing, "city"),
                    get_address_field(mailing, "stateOrCountry"),
                    get_address_field(mailing, "stateOrCountryDescription"),
                    get_address_field(mailing, "zipCode"),
                    get_address_field(business, "street1"),
                    get_address_field(business, "street2"),
                    get_address_field(business, "city"),
                    get_address_field(business, "stateOrCountry"),
                    get_address_field(business, "stateOrCountryDescription"),
                    get_address_field(business, "zipCode"),
                ])
                entity_rows += 1

                # ------------------------------------------------
                # former names
                # ------------------------------------------------
                if isinstance(former_names, list):
                    for item in former_names:
                        if not isinstance(item, dict):
                            continue
                        former_writer.writerow([
                            rel_path,
                            cik,
                            "" if item.get("name") is None else str(item.get("name")),
                            "" if item.get("from") is None else str(item.get("from")),
                            "" if item.get("to") is None else str(item.get("to")),
                        ])
                        former_name_rows += 1

                # ------------------------------------------------
                # filings.files
                # ------------------------------------------------
                if isinstance(filing_files, list):
                    for item in filing_files:
                        if not isinstance(item, dict):
                            continue
                        file_writer.writerow([
                            rel_path,
                            cik,
                            "" if item.get("name") is None else str(item.get("name")),
                            "" if item.get("filingCount") is None else str(item.get("filingCount")),
                            "" if item.get("filingFrom") is None else str(item.get("filingFrom")),
                            "" if item.get("filingTo") is None else str(item.get("filingTo")),
                        ])
                        filing_file_rows += 1

                # ------------------------------------------------
                # filings.recent
                # ------------------------------------------------
                n_recent_filings = 0
                earliest_filing_date = ""
                latest_filing_date = ""

                if isinstance(recent, dict) and recent:
                    lengths = [len(v) for v in recent.values() if isinstance(v, list)]
                    n_recent_filings = max(lengths) if lengths else 0

                    accession_numbers = recent.get("accessionNumber", [])
                    filing_dates = recent.get("filingDate", [])
                    report_dates = recent.get("reportDate", [])
                    acceptance_times = recent.get("acceptanceDateTime", [])
                    acts = recent.get("act", [])
                    forms = recent.get("form", [])
                    file_numbers = recent.get("fileNumber", [])
                    film_numbers = recent.get("filmNumber", [])
                    items = recent.get("items", [])
                    sizes = recent.get("size", [])
                    is_xbrl = recent.get("isXBRL", [])
                    is_inline_xbrl = recent.get("isInlineXBRL", [])
                    primary_docs = recent.get("primaryDocument", [])
                    primary_doc_descs = recent.get("primaryDocDescription", [])

                    for i in range(n_recent_filings):
                        filing_date = safe_list_item(filing_dates, i)
                        earliest_filing_date = safe_min_date(earliest_filing_date, filing_date)
                        latest_filing_date = safe_max_date(latest_filing_date, filing_date)

                        recent_writer.writerow([
                            rel_path,
                            cik,
                            safe_list_item(accession_numbers, i),
                            filing_date,
                            safe_list_item(report_dates, i),
                            safe_list_item(acceptance_times, i),
                            safe_list_item(acts, i),
                            safe_list_item(forms, i),
                            safe_list_item(file_numbers, i),
                            safe_list_item(film_numbers, i),
                            safe_list_item(items, i),
                            safe_list_item(sizes, i),
                            safe_list_item(is_xbrl, i),
                            safe_list_item(is_inline_xbrl, i),
                            safe_list_item(primary_docs, i),
                            safe_list_item(primary_doc_descs, i),
                        ])
                        recent_rows += 1

                # ------------------------------------------------
                # inventory row
                # ------------------------------------------------
                inv_writer.writerow([
                    rel_path,
                    file_size,
                    cik,
                    entity_name,
                    entity_type,
                    n_tickers,
                    n_exchanges,
                    n_former_names,
                    n_recent_filings,
                    n_file_refs,
                    earliest_filing_date,
                    latest_filing_date,
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
                    "",
                    0,
                    0,
                    0,
                    0,
                    0,
                    "",
                    "",
                    "error",
                    str(exc),
                ])
                inventory_rows += 1

        inv_fh.flush()
        err_fh.flush()
        ent_fh.flush()
        recent_fh.flush()
        file_fh.flush()
        former_fh.flush()

    return {
        "shard_id": shard_id,
        "inventory_part": str(inventory_part_path),
        "errors_part": str(errors_part_path),
        "entities_part": str(entities_part_path),
        "recent_filings_part": str(recent_filings_part_path),
        "filing_files_part": str(filing_files_part_path),
        "former_names_part": str(former_names_part_path),
        "input_files": len(files),
        "ok_files": ok_files,
        "failed_files": failed_files,
        "inventory_rows": inventory_rows,
        "error_rows": error_rows,
        "entity_rows": entity_rows,
        "recent_rows": recent_rows,
        "filing_file_rows": filing_file_rows,
        "former_name_rows": former_name_rows,
    }


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory + flatten SEC submissions JSON files into partitioned CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(RAW_INPUT_DIR),
        help="Directory containing extracted SEC submissions JSON files.",
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
        help="Approximate shards = workers * shard_multiplier.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous processed submissions outputs before running.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip files already recorded in existing inventory or inventory shards.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=8.0,
        help="Minimum free disk space to keep available. If free space drops below this, stop gracefully.",
    )
    parser.add_argument(
        "--disk-check-every-files",
        type=int,
        default=32,
        help="Worker disk space check frequency (every N files).",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    workers = choose_worker_count(args.workers)
    min_free_bytes = int(args.min_free_gb * (1024 ** 3))

    ensure_dirs(args.overwrite)

    print(f"Input directory : {input_dir}", flush=True)
    print(f"Output root     : {OUT_ROOT}", flush=True)
    print(f"JSON parser     : {'orjson' if HAS_ORJSON else 'stdlib json'}", flush=True)
    print(f"Workers         : {workers}", flush=True)
    print(f"Resume mode     : {args.resume}", flush=True)
    print(f"Min free space  : {args.min_free_gb:.2f} GB", flush=True)

    overall_start = now_ts()

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------
    discover_start = now_ts()
    all_files = discover_json_files(input_dir)
    all_input_bytes = sum(p.stat().st_size for p in all_files)
    discover_elapsed = now_ts() - discover_start

    print(
        f"Discovered {len(all_files):,} submissions JSON files "
        f"({human_bytes(all_input_bytes)}) in {fmt_elapsed(discover_elapsed)}",
        flush=True,
    )

    current_free = free_bytes(OUT_ROOT)
    print(f"Current free disk: {human_bytes(current_free)}", flush=True)
    if current_free < min_free_bytes:
        raise SystemExit(
            f"Refusing to start: free disk space {human_bytes(current_free)} is already below "
            f"the required minimum of {human_bytes(min_free_bytes)}."
        )

    # --------------------------------------------------------
    # Incremental resume filtering
    # --------------------------------------------------------
    previous_summary = load_previous_summary(SUMMARY_FINAL)

    processed_relpaths: set[str] = set()
    processed_basenames: set[str] = set()
    files = all_files
    skipped_existing = 0

    if args.resume:
        processed_relpaths, processed_basenames = load_existing_processed_keys()

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
        print("No new submissions JSON files to process. Nothing to do.", flush=True)
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
    total_inventory_rows = 0
    total_error_rows = 0
    total_entity_rows = 0
    total_recent_rows = 0
    total_filing_file_rows = 0
    total_former_name_rows = 0

    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        for offset, shard_files in enumerate(shards):
            shard_id = first_part_id + offset
            futures.append(
                executor.submit(
                    process_shard,
                    shard_id,
                    [str(p) for p in shard_files],
                    str(input_dir),
                    str(PARTS_DIR),
                    min_free_bytes,
                    args.disk_check_every_files,
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
            total_inventory_rows += result["inventory_rows"]
            total_error_rows += result["error_rows"]
            total_entity_rows += result["entity_rows"]
            total_recent_rows += result["recent_rows"]
            total_filing_file_rows += result["filing_file_rows"]
            total_former_name_rows += result["former_name_rows"]

            elapsed = max(now_ts() - shard_start, 1e-9)
            files_per_sec = total_files_done / elapsed
            recent_rows_per_sec = total_recent_rows / elapsed if total_recent_rows else 0.0

            print(
                f"[process] shards {completed}/{total_shards} | "
                f"new files {total_files_done:,}/{len(files):,} | "
                f"ok {total_ok_files:,} | failed {total_failed_files:,} | "
                f"recent rows {total_recent_rows:,} | "
                f"{files_per_sec:,.1f} files/s | {recent_rows_per_sec:,.1f} rows/s",
                flush=True,
            )

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Partial shard outputs are preserved. "
            "Re-run with --resume to continue from completed work.",
            flush=True,
        )
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    except LowDiskSpaceError as exc:
        print(
            f"\nStopped due to low disk space: {exc}\n"
            f"Partial shard outputs are preserved. Make space and re-run with --resume.",
            flush=True,
        )
        executor.shutdown(wait=False, cancel_futures=True)
        return

    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    finally:
        try:
            executor.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    process_elapsed = now_ts() - shard_start

    # --------------------------------------------------------
    # Merge inventory and errors only
    # --------------------------------------------------------
    inventory_parts = sorted(PARTS_DIR.glob("inventory_part_*.csv"))
    error_parts = sorted(PARTS_DIR.glob("errors_part_*.csv"))

    print("Rebuilding cumulative inventory CSV...", flush=True)
    merge_csv_parts(inventory_parts, INVENTORY_FINAL, "[merge inventory]")

    print("Rebuilding cumulative errors CSV...", flush=True)
    merge_csv_parts(error_parts, ERRORS_FINAL, "[merge errors]")

    # --------------------------------------------------------
    # Other outputs stay partitioned
    # --------------------------------------------------------
    entities_parts = sorted(PARTS_DIR.glob("entities_part_*.csv"))
    recent_parts = sorted(PARTS_DIR.glob("recent_filings_part_*.csv"))
    file_ref_parts = sorted(PARTS_DIR.glob("filing_files_part_*.csv"))
    former_name_parts = sorted(PARTS_DIR.glob("former_names_part_*.csv"))

    previous_total_recent_rows = int(previous_summary.get("total_recent_rows", 0) or 0)
    previous_total_entity_rows = int(previous_summary.get("total_entity_rows", 0) or 0)
    previous_total_filing_file_rows = int(previous_summary.get("total_filing_file_rows", 0) or 0)
    previous_total_former_name_rows = int(previous_summary.get("total_former_name_rows", 0) or 0)

    summary = {
        "input_dir": str(input_dir),
        "output_root": str(OUT_ROOT),
        "used_orjson": HAS_ORJSON,
        "workers": workers,
        "resume_mode": args.resume,
        "min_free_gb": args.min_free_gb,
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
        "total_entity_rows_this_run": total_entity_rows,
        "total_recent_rows_this_run": total_recent_rows,
        "total_filing_file_rows_this_run": total_filing_file_rows,
        "total_former_name_rows_this_run": total_former_name_rows,
        "total_entity_rows": previous_total_entity_rows + total_entity_rows,
        "total_recent_rows": previous_total_recent_rows + total_recent_rows,
        "total_filing_file_rows": previous_total_filing_file_rows + total_filing_file_rows,
        "total_former_name_rows": previous_total_former_name_rows + total_former_name_rows,
        "entities_parts": [str(p) for p in entities_parts],
        "recent_filings_parts": [str(p) for p in recent_parts],
        "filing_files_parts": [str(p) for p in file_ref_parts],
        "former_names_parts": [str(p) for p in former_name_parts],
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

    print("\n=== SUBMISSIONS PIPELINE COMPLETE ===", flush=True)
    print(f"Inventory CSV    : {INVENTORY_FINAL}", flush=True)
    print(f"Errors CSV       : {ERRORS_FINAL}", flush=True)
    print(f"Entity parts     : {len(entities_parts)} file(s)", flush=True)
    print(f"Recent parts     : {len(recent_parts)} file(s)", flush=True)
    print(f"File-ref parts   : {len(file_ref_parts)} file(s)", flush=True)
    print(f"Former-name parts: {len(former_name_parts)} file(s)", flush=True)
    print(f"Summary JSON     : {SUMMARY_FINAL}", flush=True)
    print(f"All visible JSON : {len(all_files):,}", flush=True)
    print(f"Processed this run: {len(files):,}", flush=True)
    print(f"Skipped existing : {skipped_existing:,}", flush=True)
    print(f"Recent filing rows this run : {total_recent_rows:,}", flush=True)
    print(f"Cumulative recent rows      : {previous_total_recent_rows + total_recent_rows:,}", flush=True)
    print(f"Elapsed           : {fmt_elapsed(total_elapsed)}", flush=True)


if __name__ == "__main__":
    main()

# python data/sec_submissions_pipeline.py --workers 8 --min-free-gb 8
# Resume after interruption / low disk:
# python data/sec_submissions_pipeline.py --workers 8 --resume --min-free-gb 8