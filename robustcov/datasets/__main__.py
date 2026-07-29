# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Command-line access to robustcov external dataset loaders."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from . import DATASET_REGISTRY, fetch_cmapss, fetch_gas_sensor_drift, get_data_home


def _print_info(name: str) -> None:
    info = DATASET_REGISTRY[name]
    payload = asdict(info)
    payload["default_cache_root"] = str(get_data_home())
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list supported optional datasets")
    info_parser = subparsers.add_parser("info", help="show dataset source and terms metadata")
    info_parser.add_argument("dataset", choices=tuple(DATASET_REGISTRY))

    fetch_parser = subparsers.add_parser("fetch", help="download and validate one dataset")
    fetch_parser.add_argument("dataset", choices=tuple(DATASET_REGISTRY))
    fetch_parser.add_argument("--cache-dir")
    fetch_parser.add_argument("--archive-path", help="use a manually downloaded ZIP instead")
    fetch_parser.add_argument("--subset", default="FD002", choices=("FD001", "FD002", "FD003", "FD004"))

    args = parser.parse_args()
    if args.command == "list":
        for name, info in DATASET_REGISTRY.items():
            print(f"{name}\t{info.name}\t{info.homepage}")
        return
    if args.command == "info":
        _print_info(args.dataset)
        return

    if args.dataset == "gas_sensor_drift":
        dataset = fetch_gas_sensor_drift(
            cache_dir=args.cache_dir,
            download=args.archive_path is None,
            archive_path=args.archive_path,
        )
        print(f"dataset,{dataset.info.name}")
        print(f"shape,{dataset.X.shape[0]},{dataset.X.shape[1]}")
        print(f"batches,{','.join(map(str, sorted(set(dataset.batch.tolist()))))}")
    else:
        dataset = fetch_cmapss(
            args.subset,
            cache_dir=args.cache_dir,
            download=args.archive_path is None,
            archive_path=args.archive_path,
        )
        print(f"dataset,{dataset.info.name}")
        print(f"subset,{dataset.subset}")
        print(f"train_rows,{dataset.train.sensors.shape[0]}")
        print(f"test_rows,{dataset.test.sensors.shape[0]}")
    print(f"cache_dir,{dataset.data_dir.parent}")
    print(f"archive_sha256,{dataset.archive_sha256}")


if __name__ == "__main__":
    main()
