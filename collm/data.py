"""Rebuild the original time split from official MovieLens files.

History is rolling observed positive feedback: earlier validation/test feedback may
appear in later prompts, but never in MF fitting or model parameter updates.
Events sharing a timestamp are processed as a group so they cannot see each other.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .common import ROOT, sha256, write_json


def make_records(frame, history_limit=10):
    if not isinstance(history_limit, int) or isinstance(history_limit, bool) or history_limit < 1:
        raise ValueError("history_limit must be a positive integer")
    frame = frame.sort_values(["uid", "timestamp", "iid"], kind="stable")
    users = {int(v): n + 1 for n, v in enumerate(sorted(frame.uid.unique()))}
    items = {int(v): n + 1 for n, v in enumerate(sorted(frame.iid.unique()))}
    output = {s: [] for s in ("train", "valid", "test")}
    history = defaultdict(list)
    # Group equal-time events: a prompt never contains a same-time/future rating.
    for (raw_uid, timestamp), group in frame.groupby(["uid", "timestamp"], sort=False):
        hist = history[raw_uid][-history_limit:]
        for row in group.itertuples(index=False):
            output[row.split].append(
                {
                    "record_id": f"{int(row.uid)}:{int(row.iid)}:{int(row.timestamp)}",
                    "uid": users[int(row.uid)],
                    "iid": items[int(row.iid)],
                    "raw_uid": int(row.uid),
                    "raw_iid": int(row.iid),
                    "timestamp": int(timestamp),
                    "label": int(row.rating > 3),
                    "title": row.title,
                    "history_titles": [title for title, _ in hist],
                    "history_timestamps": [time for _, time in hist],
                }
            )
        for row in group.itertuples(index=False):
            if row.rating > 3:
                history[raw_uid].append((row.title, int(timestamp)))
        history[raw_uid] = history[raw_uid][-history_limit:]
    return output, users, items


def prepare():
    from .assets import verify_dataset_archive

    archive = ROOT / "data/raw/ml-1m.zip"
    verify_dataset_archive(archive)
    with zipfile.ZipFile(archive) as z:
        ratings = pd.read_csv(
            io.BytesIO(z.read("ml-1m/ratings.dat")),
            sep="::",
            engine="python",
            names=["uid", "iid", "rating", "timestamp"],
        )
        movies = pd.read_csv(
            io.BytesIO(z.read("ml-1m/movies.dat")),
            sep="::",
            engine="python",
            encoding="latin1",
            names=["iid", "title", "genres"],
        )
    dates = pd.to_datetime(ratings.timestamp, unit="s", utc=True)
    months = dates.dt.year * 12 + dates.dt.month
    ratings["month_slot"] = months - months.min()
    selected = ratings[ratings.month_slot.between(14, 33)].copy()
    selected["split"] = np.select(
        [selected.month_slot < 24, selected.month_slot < 29], ["train", "valid"], default="test"
    )
    selected = selected.merge(movies[["iid", "title"]], on="iid", validate="many_to_one")
    records, users, items = make_records(selected)
    root = ROOT / "data/processed"
    root.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, rows in records.items():
        path = root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] = {
            "count": len(rows),
            "positive_rate": np.mean([r["label"] for r in rows]),
            "min_timestamp": min(r["timestamp"] for r in rows),
            "max_timestamp": max(r["timestamp"] for r in rows),
            "sha256": sha256(path),
        }
    metadata = {
        "num_users": len(users) + 1,
        "num_items": len(items) + 1,
        "id_mapping": "sorted raw IDs across retained splits, 0 reserved; no labels used",
        "label": "rating > 3",
        "month_slots": {"train": [14, 23], "valid": [24, 28], "test": [29, 33]},
        "history_limit": 10,
        "history_protocol": "rolling observed positives, strictly earlier timestamps",
        "history_difference_from_upstream": "same-timestamp events cannot enter each other's histories",
        "splits": stats,
        "archive_sha256": sha256(archive),
    }
    write_json(root / "metadata.json", metadata)
    write_json(ROOT / "results/data_summary.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def load_rows(split, limit=None, seed=42):
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError(f"{split}_limit must be a positive integer or null for the full split")
    path = ROOT / "data/processed" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if limit is not None and limit < len(rows):
        chosen = np.random.default_rng(seed).choice(len(rows), size=limit, replace=False)
        rows = [rows[int(i)] for i in sorted(chosen)]
    if len({r["label"] for r in rows}) != 2:
        raise ValueError(f"{split} must contain both label classes for ROC AUC")
    return rows


def experiment_data(cfg):
    rows = {
        s: load_rows(s, cfg["data"].get(f"{s}_limit"), cfg["seed"])
        for s in ("train", "valid", "test")
    }
    train_users = Counter(r["uid"] for r in rows["train"])
    train_items = Counter(r["iid"] for r in rows["train"])
    for split_rows in rows.values():
        for row in split_rows:
            row["warm"] = train_users[row["uid"]] >= 3 and train_items[row["iid"]] >= 3
    return rows


def split_fingerprint(rows):
    import hashlib

    return hashlib.sha256("\n".join(r["record_id"] for r in rows).encode()).hexdigest()


def manifest(rows):
    import hashlib

    def content_fingerprint(records):
        canonical = "\n".join(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
            for row in records
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return {
        s: {
            "count": len(v),
            "record_ids_sha256": split_fingerprint(v),
            "content_sha256": content_fingerprint(v),
            "positive_rate": float(np.mean([r["label"] for r in v])),
        }
        for s, v in rows.items()
    }


if __name__ == "__main__":
    prepare()
