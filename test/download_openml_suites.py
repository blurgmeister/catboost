#!/usr/bin/env python3
"""
Download a practical subset of OpenML-CC18, OpenML-CTR23, and selected
standalone OpenML benchmark datasets as CSV files.

The script uses OpenML benchmark suites rather than hard-coded dataset IDs:
  - OpenML-CC18: curated classification benchmark, suite id 99
  - OpenML-CTR23: curated tabular regression benchmark, suite id 353

It also downloads selected standalone OpenML datasets:
  - French Motor Claims freMTPL2freq: claim frequency dataset, data id 43593
  - Covertype: forest cover type classification dataset, data id 1596

Each CSV contains feature columns plus a final target column named "__target__".
Metadata for every saved dataset is written to manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import openml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: openml. Install it with:\n"
        "  python -m pip install openml pandas scikit-learn pyarrow\n"
    ) from exc


OUTPUT_DIR = Path("/home/kerith/workspaces/smooth_multi")
TARGET_COLUMN = "__target__"
FRENCH_MOTOR_CLAIMS_FREQ = {
    "suite": "OpenML-French-Motor-Claims",
    "dataset_id": 43593,
    "dataset_name": "French-Motor-Claims-Datasets-freMTPL2freq",
    "target_name": "ClaimNb",
}
STANDALONE_DATASETS = {
    "french-motor-claims": FRENCH_MOTOR_CLAIMS_FREQ,
    "covertype": {
        "suite": "OpenML-Covertype",
        "dataset_id": 1596,
        "dataset_name": "covertype",
        "target_name": "class",
    },
}


@dataclass(frozen=True)
class Candidate:
    suite: str
    task_id: int
    dataset_id: int
    dataset_name: str
    target_name: str
    n_rows: int
    n_features: int
    n_classes: int | None


@dataclass(frozen=True)
class TaskCandidate:
    suite: str
    task_id: int


@dataclass(frozen=True)
class DatasetCandidate:
    suite: str
    dataset_id: int
    dataset_name: str
    target_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected OpenML benchmark datasets as CSV files."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--only-standalone",
        action="store_true",
        help="Skip OpenML-CC18/OpenML-CTR23 suites and only download standalone datasets.",
    )
    parser.add_argument(
        "--standalone-dataset",
        action="append",
        choices=sorted(STANDALONE_DATASETS),
        default=None,
        help=(
            "Standalone dataset to download. May be passed multiple times. "
            f"Defaults to all: {', '.join(sorted(STANDALONE_DATASETS))}."
        ),
    )
    parser.add_argument("--cc18-count", type=int, default=12)
    parser.add_argument("--ctr23-count", type=int, default=8)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets from both suites, ignoring --cc18-count and --ctr23-count.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="Skip datasets larger than this row count.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5_000,
        help="Skip datasets larger than this feature count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite CSV files that already exist.",
    )
    parser.add_argument(
        "--metadata-select",
        action="store_true",
        help="Query every task's metadata before selecting datasets. Slower, but more size-balanced.",
    )
    parser.add_argument(
        "--openml-cache-dir",
        type=Path,
        default=None,
        help="Optional OpenML cache directory.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value[:120] or "dataset"


def quality_as_int(qualities: dict, *names: str, default: int = 0) -> int:
    for name in names:
        value = qualities.get(name)
        if value is not None and value == value:
            return int(float(value))
    return default


def get_suite(alias: str, fallback_id: int):
    try:
        print(f"Fetching OpenML suite {alias}...", flush=True)
        return openml.study.get_suite(alias)
    except Exception:
        print(f"Alias lookup failed; fetching OpenML suite id {fallback_id}...", flush=True)
        return openml.study.get_suite(fallback_id)


def collect_candidates(
    suite_alias: str,
    suite_id: int,
    max_rows: int,
    max_features: int,
) -> list[Candidate]:
    suite = get_suite(suite_alias, suite_id)
    candidates: list[Candidate] = []
    print(f"{suite_alias}: found {len(suite.tasks)} tasks; collecting metadata...", flush=True)
    for idx, task_id in enumerate(suite.tasks, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(suite.tasks):
            print(f"{suite_alias}: metadata {idx}/{len(suite.tasks)}", flush=True)
        try:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            qualities = dataset.qualities or {}
            n_rows = quality_as_int(qualities, "NumberOfInstances", "number_instances")
            n_features = quality_as_int(qualities, "NumberOfFeatures", "number_features")
            n_classes = quality_as_int(qualities, "NumberOfClasses", "number_classes", default=-1)
            if n_rows > max_rows or n_features > max_features:
                continue
            candidates.append(
                Candidate(
                    suite=suite_alias,
                    task_id=int(task_id),
                    dataset_id=int(dataset.dataset_id),
                    dataset_name=str(dataset.name),
                    target_name=str(task.target_name),
                    n_rows=n_rows,
                    n_features=n_features,
                    n_classes=None if n_classes < 0 else n_classes,
                )
            )
        except Exception as exc:
            print(f"Skipping task {task_id} from {suite_alias}: {exc}", file=sys.stderr)
    return candidates


def size_bucket(candidate: Candidate) -> int:
    if candidate.n_rows < 2_000:
        return 0
    if candidate.n_rows < 10_000:
        return 1
    return 2


def select_diverse(candidates: list[Candidate], count: int, classification: bool) -> list[Candidate]:
    if count >= len(candidates):
        return sorted(candidates, key=lambda c: (c.n_rows, c.n_features, c.task_id))

    ordered = sorted(candidates, key=lambda c: (size_bucket(c), c.n_rows, c.n_features, c.task_id))
    selected: list[Candidate] = []
    used: set[int] = set()

    groups: list[list[Candidate]] = []
    for bucket in (0, 1, 2):
        bucket_items = [c for c in ordered if size_bucket(c) == bucket]
        if classification:
            groups.append([c for c in bucket_items if c.n_classes == 2])
            groups.append([c for c in bucket_items if c.n_classes and c.n_classes > 2])
        else:
            groups.append(bucket_items)

    while len(selected) < count and any(groups):
        next_groups: list[list[Candidate]] = []
        for group in groups:
            if not group:
                continue
            candidate = group.pop(len(group) // 2)
            if candidate.task_id not in used:
                selected.append(candidate)
                used.add(candidate.task_id)
                if len(selected) == count:
                    break
            if group:
                next_groups.append(group)
        groups = next_groups

    if len(selected) < count:
        for candidate in ordered:
            if candidate.task_id not in used:
                selected.append(candidate)
                used.add(candidate.task_id)
                if len(selected) == count:
                    break

    return sorted(selected, key=lambda c: (c.suite, c.task_id))


def select_evenly(task_ids: list[int], count: int) -> list[int]:
    if count >= len(task_ids):
        return list(task_ids)
    if count <= 0:
        return []
    if count == 1:
        return [task_ids[len(task_ids) // 2]]
    indexes = [round(i * (len(task_ids) - 1) / (count - 1)) for i in range(count)]
    selected: list[int] = []
    seen: set[int] = set()
    for index in indexes:
        task_id = int(task_ids[index])
        if task_id not in seen:
            selected.append(task_id)
            seen.add(task_id)
    for task_id in task_ids:
        task_id = int(task_id)
        if len(selected) == count:
            break
        if task_id not in seen:
            selected.append(task_id)
            seen.add(task_id)
    return selected


def download_candidate(candidate: Candidate, output_dir: Path, overwrite: bool) -> dict:
    suite_dir = output_dir / candidate.suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{candidate.suite}_task-{candidate.task_id}_data-{candidate.dataset_id}_"
        f"{safe_name(candidate.dataset_name)}.csv"
    )
    csv_path = suite_dir / filename

    if csv_path.exists() and not overwrite:
        print(f"  exists: {csv_path}", flush=True)
    else:
        task = openml.tasks.get_task(candidate.task_id)
        dataset = task.get_dataset()
        print(f"  downloading data...", flush=True)
        X, y, _, _ = dataset.get_data(
            target=task.target_name,
            dataset_format="dataframe",
        )
        frame = X.copy()
        frame[TARGET_COLUMN] = pd.Series(y).reset_index(drop=True)
        print(f"  writing {csv_path}", flush=True)
        frame.to_csv(csv_path, index=False)

    return {
        "suite": candidate.suite,
        "task_id": candidate.task_id,
        "dataset_id": candidate.dataset_id,
        "dataset_name": candidate.dataset_name,
        "target_original": candidate.target_name,
        "target_csv": TARGET_COLUMN,
        "n_rows": candidate.n_rows,
        "n_features": candidate.n_features,
        "n_classes": "" if candidate.n_classes is None else candidate.n_classes,
        "csv_path": str(csv_path),
    }


def download_task_candidate(candidate: TaskCandidate, output_dir: Path, overwrite: bool) -> dict | None:
    print(f"Fetching task {candidate.task_id}...", flush=True)
    try:
        task = openml.tasks.get_task(candidate.task_id)
        dataset = task.get_dataset()
        dataset_id = int(dataset.dataset_id)
        dataset_name = str(dataset.name)
        target_name = str(task.target_name)
        suite_dir = output_dir / candidate.suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        csv_path = suite_dir / (
            f"{candidate.suite}_task-{candidate.task_id}_data-{dataset_id}_"
            f"{safe_name(dataset_name)}.csv"
        )

        if csv_path.exists() and not overwrite:
            print(f"  exists: {csv_path}", flush=True)
            frame = pd.read_csv(csv_path, nrows=5)
            n_features = max(0, len(frame.columns) - 1)
            n_rows = ""
        else:
            print(f"  downloading {dataset_name}...", flush=True)
            X, y, _, _ = dataset.get_data(target=target_name, dataset_format="dataframe")
            frame = X.copy()
            frame[TARGET_COLUMN] = pd.Series(y).reset_index(drop=True)
            n_features = X.shape[1]
            n_rows = X.shape[0]
            print(f"  writing {csv_path}", flush=True)
            frame.to_csv(csv_path, index=False)

        return {
            "suite": candidate.suite,
            "task_id": candidate.task_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "target_original": target_name,
            "target_csv": TARGET_COLUMN,
            "n_rows": n_rows,
            "n_features": n_features,
            "n_classes": "",
            "csv_path": str(csv_path),
        }
    except Exception as exc:
        print(f"Skipping {candidate.suite} task {candidate.task_id}: {exc}", file=sys.stderr, flush=True)
        return None


def download_dataset_candidate(candidate: DatasetCandidate, output_dir: Path, overwrite: bool) -> dict | None:
    print(f"Fetching dataset {candidate.dataset_id} ({candidate.dataset_name})...", flush=True)
    try:
        dataset = openml.datasets.get_dataset(candidate.dataset_id)
        dataset_name = str(dataset.name or candidate.dataset_name)
        suite_dir = output_dir / candidate.suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        csv_path = suite_dir / (
            f"{candidate.suite}_data-{candidate.dataset_id}_"
            f"{safe_name(dataset_name)}.csv"
        )

        if csv_path.exists() and not overwrite:
            print(f"  exists: {csv_path}", flush=True)
            frame = pd.read_csv(csv_path, nrows=5)
            n_features = max(0, len(frame.columns) - 1)
            n_rows = ""
        else:
            print(f"  downloading {dataset_name}...", flush=True)
            X, y, _, _ = dataset.get_data(
                target=candidate.target_name,
                dataset_format="dataframe",
            )
            frame = X.copy()
            frame[TARGET_COLUMN] = pd.Series(y).reset_index(drop=True)
            n_features = X.shape[1]
            n_rows = X.shape[0]
            print(f"  writing {csv_path}", flush=True)
            frame.to_csv(csv_path, index=False)

        return {
            "suite": candidate.suite,
            "task_id": "",
            "dataset_id": candidate.dataset_id,
            "dataset_name": dataset_name,
            "target_original": candidate.target_name,
            "target_csv": TARGET_COLUMN,
            "n_rows": n_rows,
            "n_features": n_features,
            "n_classes": "",
            "csv_path": str(csv_path),
        }
    except Exception as exc:
        print(
            f"Skipping {candidate.suite} dataset {candidate.dataset_id}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None


def select_standalone_datasets(names: list[str] | None) -> list[DatasetCandidate]:
    selected_names = names or sorted(STANDALONE_DATASETS)
    return [
        DatasetCandidate(
            suite=STANDALONE_DATASETS[name]["suite"],
            dataset_id=STANDALONE_DATASETS[name]["dataset_id"],
            dataset_name=STANDALONE_DATASETS[name]["dataset_name"],
            target_name=STANDALONE_DATASETS[name]["target_name"],
        )
        for name in selected_names
    ]


def write_manifest(output_dir: Path, rows: Iterable[dict], merge_existing: bool = False) -> None:
    rows = list(rows)
    manifest_path = output_dir / "manifest.csv"
    if merge_existing and manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
        if not rows:
            print(f"No new datasets downloaded; preserving existing manifest at {manifest_path}")
            return
        replacement_keys = {(str(row["suite"]), str(row["dataset_id"])) for row in rows}
        rows = [
            row
            for row in existing_rows
            if (str(row.get("suite", "")), str(row.get("dataset_id", ""))) not in replacement_keys
        ] + rows
    if not rows:
        manifest_path.write_text("", encoding="utf-8")
        return
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.openml_cache_dir:
        args.openml_cache_dir.mkdir(parents=True, exist_ok=True)
        openml.config.set_root_cache_directory(str(args.openml_cache_dir))

    manifest_rows = []

    if args.only_standalone:
        print("Skipping OpenML-CC18 and OpenML-CTR23 suites (--only-standalone).", flush=True)
    elif args.metadata_select:
        cc18 = collect_candidates("OpenML-CC18", 99, args.max_rows, args.max_features)
        ctr23 = collect_candidates("OpenML-CTR23", 353, args.max_rows, args.max_features)

        if args.all:
            selected = sorted(cc18 + ctr23, key=lambda c: (c.suite, c.task_id))
        else:
            selected = (
                select_diverse(cc18, args.cc18_count, classification=True)
                + select_diverse(ctr23, args.ctr23_count, classification=False)
            )

        for idx, candidate in enumerate(selected, start=1):
            print(
                f"[{idx}/{len(selected)}] {candidate.suite} task={candidate.task_id} "
                f"data={candidate.dataset_id} rows={candidate.n_rows} "
                f"features={candidate.n_features} name={candidate.dataset_name}",
                flush=True,
            )
            manifest_rows.append(download_candidate(candidate, args.output_dir, args.overwrite))
    else:
        cc18_suite = get_suite("OpenML-CC18", 99)
        ctr23_suite = get_suite("OpenML-CTR23", 353)
        cc18_task_ids = [int(task_id) for task_id in cc18_suite.tasks]
        ctr23_task_ids = [int(task_id) for task_id in ctr23_suite.tasks]
        if args.all:
            selected_tasks = (
                [TaskCandidate("OpenML-CC18", task_id) for task_id in cc18_task_ids]
                + [TaskCandidate("OpenML-CTR23", task_id) for task_id in ctr23_task_ids]
            )
        else:
            selected_tasks = (
                [TaskCandidate("OpenML-CC18", task_id) for task_id in select_evenly(cc18_task_ids, args.cc18_count)]
                + [TaskCandidate("OpenML-CTR23", task_id) for task_id in select_evenly(ctr23_task_ids, args.ctr23_count)]
            )

        for idx, candidate in enumerate(selected_tasks, start=1):
            print(f"[{idx}/{len(selected_tasks)}] {candidate.suite} task={candidate.task_id}", flush=True)
            row = download_task_candidate(candidate, args.output_dir, args.overwrite)
            if row is not None:
                manifest_rows.append(row)

    standalone_datasets = select_standalone_datasets(args.standalone_dataset)
    for idx, candidate in enumerate(standalone_datasets, start=1):
        print(
            f"[standalone {idx}/{len(standalone_datasets)}] "
            f"{candidate.suite} data={candidate.dataset_id}",
            flush=True,
        )
        row = download_dataset_candidate(candidate, args.output_dir, args.overwrite)
        if row is not None:
            manifest_rows.append(row)

    write_manifest(args.output_dir, manifest_rows, merge_existing=args.only_standalone)
    print(f"Wrote {len(manifest_rows)} datasets under {args.output_dir}")
    print(f"Wrote manifest to {args.output_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
