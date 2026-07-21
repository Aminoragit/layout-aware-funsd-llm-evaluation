#!/usr/bin/env python3
"""Run the real-document FUNSD LLM entity-classification experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import struct
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "funsd_dataset.zip"
MANIFEST = ROOT / "source_manifest.json"
RAW = ROOT / "raw_model_outputs.jsonl"
PER_PAGE = ROOT / "per_page_metrics.csv"
AGGREGATE = ROOT / "aggregate_results.json"
LOG = ROOT / "execution.log"

ENDPOINT = os.environ.get("PAPER2_LLM_URL", "http://127.0.0.1:11440/v1").rstrip("/")
MODEL = "openai/qwable-3.6-27b-mtp"
SEEDS = [17, 29, 43]
TEMPERATURE = 0.2
TOP_P = 0.9
LABELS = ["header", "question", "answer", "other"]
SAMPLE_IDS = ["82092117", "82200067_0069", "82250337_0338", "82251504"]
EXPECTED_SHA256 = "c31735649e4f441bcbb4fd0f379574f7520b42286e80b01d80b445649d54761f"
MAX_GENERATION_ATTEMPTS = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("FUNSD source image is not PNG")
    return struct.unpack(">II", data[16:24])


def load_pages() -> list[dict]:
    if not DATASET.exists():
        raise SystemExit(
            "Missing artifacts/funsd_dataset.zip. Read DATASET_LICENSE_NOTICE.md, accept the FUNSD terms, "
            "then download https://guillaumejaume.github.io/FUNSD/dataset.zip to that path."
        )
    digest = sha256(DATASET)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"FUNSD archive SHA-256 mismatch: {digest}")
    pages = []
    with zipfile.ZipFile(DATASET) as z:
        for page_id in SAMPLE_IDS:
            annotation_path = f"dataset/testing_data/annotations/{page_id}.json"
            image_path = f"dataset/testing_data/images/{page_id}.png"
            annotation_bytes = z.read(annotation_path)
            image_bytes = z.read(image_path)
            width, height = png_size(image_bytes)
            entities, excluded_empty = [], 0
            for item in json.loads(annotation_bytes)["form"]:
                text = item.get("text", "").strip()
                if not text:
                    excluded_empty += 1
                    continue
                entities.append({"id": int(item["id"]), "text": text, "box": item["box"], "label": item["label"]})
            pages.append(
                {
                    "page_id": page_id,
                    "width": width,
                    "height": height,
                    "entities": sorted(entities, key=lambda x: x["id"]),
                    "excluded_empty": excluded_empty,
                    "annotation_path": annotation_path,
                    "image_path": image_path,
                    "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                }
            )
    return pages


SYSTEM = """You classify OCR entities from scanned business forms. Assign exactly one FUNSD label to every entity: header (section/document heading), question (field name or prompt), answer (filled value), or other (instructions, boilerplate, marks, or uncategorized text). Return only one JSON object shaped as {\"pages\":{\"PAGE_ID\":{\"ENTITY_ID\":\"label\"}}}. Include every supplied page and entity ID once and no additional IDs."""


def make_prompt(pages: list[dict], condition: str) -> str:
    documents = []
    for page in pages:
        if condition == "text_only":
            entities = [{"id": e["id"], "text": e["text"]} for e in page["entities"]]
        else:
            entities = [
                {
                    "id": e["id"],
                    "text": e["text"],
                    "box_normalized": [
                        round(e["box"][0] / page["width"], 4),
                        round(e["box"][1] / page["height"], 4),
                        round(e["box"][2] / page["width"], 4),
                        round(e["box"][3] / page["height"], 4),
                    ],
                }
                for e in page["entities"]
            ]
        documents.append({"page_id": page["page_id"], "entities": entities})
    boundary = "OCR text only; dataset ID order; no geometry." if condition == "text_only" else (
        "OCR text plus normalized [left, top, right, bottom] boxes."
    )
    return boundary + "\nDocuments:\n" + json.dumps(documents, ensure_ascii=False, separators=(",", ":"))


def make_request(pages: list[dict], condition: str, seed: int) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": make_prompt(pages, condition)}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": seed,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


def call_model(pages: list[dict], condition: str, seed: int) -> dict:
    request_body = make_request(pages, condition, seed)
    encoded = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    started, api_response, error, attempts = time.time(), None, None, []
    for attempt in range(1, 4):
        req = urllib.request.Request(
            ENDPOINT + "/chat/completions", data=encoded, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                api_response = json.load(response)
            attempts.append({"attempt": attempt, "status": "success"})
            error = None
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "status": "error", "error": error})
            if attempt < 3:
                time.sleep(10 * attempt)
    return {
        "page_ids": SAMPLE_IDS,
        "condition": condition,
        "seed": seed,
        "request": request_body,
        "response": api_response,
        "error": error,
        "attempts": attempts,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_batch(record: dict, pages: list[dict]) -> tuple[dict[str, dict[int, str]], str | None]:
    if record.get("error"):
        return {}, record["error"]
    try:
        content = record["response"]["choices"][0]["message"].get("content") or ""
        payload = json.loads(content, object_pairs_hook=unique_object)
        if set(payload) != {"pages"} or not isinstance(payload["pages"], dict):
            raise ValueError("response must contain only a pages object")
        result = {}
        raw_pages = payload["pages"]
        expected_pages = {page["page_id"] for page in pages}
        if set(raw_pages) != expected_pages:
            raise ValueError("response page IDs do not exactly match the request")
        for page in pages:
            expected = {e["id"] for e in page["entities"]}
            raw_labels = raw_pages[page["page_id"]]
            if not isinstance(raw_labels, dict):
                raise ValueError(f"page {page['page_id']} labels must be an object")
            if set(raw_labels) != {str(entity_id) for entity_id in expected}:
                raise ValueError(f"page {page['page_id']} entity IDs do not exactly match the request")
            labels = {int(k): str(v).strip().lower() for k, v in raw_labels.items()}
            if any(label not in LABELS for label in labels.values()):
                raise ValueError(f"page {page['page_id']} contains an invalid label")
            result[page["page_id"]] = labels
        return result, None
    except Exception as exc:
        return {}, f"parse error: {type(exc).__name__}: {exc}"


def record_status(record: dict, pages: list[dict]) -> str:
    if record.get("error"):
        return "transport_error"
    response = record.get("response") or {}
    choices = response.get("choices") or []
    if not choices:
        return "invalid_response"
    if choices[0].get("finish_reason") != "stop":
        return "truncated" if choices[0].get("finish_reason") == "length" else "invalid_finish_reason"
    condition, seed = record.get("condition"), record.get("seed")
    if condition not in {"text_only", "layout_aware"} or seed not in SEEDS:
        return "invalid_metadata"
    if record.get("page_ids") != SAMPLE_IDS or record.get("request") != make_request(pages, condition, seed):
        return "nonuniform_protocol"
    if response.get("model") != MODEL:
        return "wrong_model"
    _, error = parse_batch(record, pages)
    return "selected_uniform_complete" if not error else "invalid_output"


def score(gold: dict[int, str], predicted: dict[int, str]) -> dict:
    correct = sum(predicted.get(i) == label for i, label in gold.items())
    per_label_f1 = {}
    for label in LABELS:
        tp = sum(gold[i] == label and predicted.get(i) == label for i in gold)
        fp = sum(gold[i] != label and predicted.get(i) == label for i in gold)
        fn = sum(gold[i] == label and predicted.get(i) != label for i in gold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label_f1[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "entities": len(gold),
        "correct": correct,
        "accuracy": correct / len(gold),
        "macro_f1": statistics.mean(per_label_f1.values()),
        "missing_or_invalid": len(gold) - len(predicted),
        "per_label_f1": per_label_f1,
    }


def aggregate(rows: list[dict], raw_records: list[dict]) -> dict:
    usage = {(r["condition"], r["seed"]): (r.get("response") or {}).get("usage", {}) for r in raw_records}
    trials = []
    for condition in ("text_only", "layout_aware", "hybrid"):
        for seed in SEEDS:
            selected = [r for r in rows if r["condition"] == condition and r["seed"] == seed]
            entities = sum(r["entities"] for r in selected)
            source_usage = usage.get((condition, seed), {}) if condition != "hybrid" else {}
            trials.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "pages": len(selected),
                    "entities": entities,
                    "accuracy": sum(r["correct"] for r in selected) / entities,
                    "mean_page_macro_f1": statistics.mean(r["macro_f1"] for r in selected),
                    "prompt_tokens": source_usage.get("prompt_tokens") if condition != "hybrid" else None,
                    "layout_aware_pages": sum(r["selected_source"] == "layout_aware" for r in selected),
                }
            )
    summaries = []
    for condition in ("text_only", "layout_aware", "hybrid"):
        selected = [t for t in trials if t["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "trials": 3,
                "accuracy_mean": statistics.mean(t["accuracy"] for t in selected),
                "accuracy_sd": statistics.stdev(t["accuracy"] for t in selected),
                "mean_page_macro_f1_mean": statistics.mean(t["mean_page_macro_f1"] for t in selected),
                "mean_page_macro_f1_sd": statistics.stdev(t["mean_page_macro_f1"] for t in selected),
                "prompt_tokens_mean": None if condition == "hybrid" else statistics.mean(t["prompt_tokens"] for t in selected),
                "layout_aware_pages": selected[0]["layout_aware_pages"],
            }
        )
    return {"trials": trials, "summary": summaries}


def main() -> None:
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    verify_only = "--verify-only" in sys.argv
    pages = load_pages()
    jobs = [(condition, seed) for condition in ("text_only", "layout_aware") for seed in SEEDS]
    raw_records = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()] if RAW.exists() else []
    def valid(record: dict) -> bool:
        return record_status(record, pages) == "selected_uniform_complete"

    completed = {(r["condition"], r["seed"]) for r in raw_records if valid(r)}
    for condition, seed in jobs:
        if (condition, seed) in completed:
            continue
        if verify_only:
            raise SystemExit(f"FAIL: no complete uniform output for {condition}, seed {seed}")
        for _ in range(MAX_GENERATION_ATTEMPTS):
            record = call_model(pages, condition, seed)
            raw_records.append(record)
            RAW.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_records), encoding="utf-8")
            print(condition, seed, record_status(record, pages).upper(), flush=True)
            if valid(record):
                break

    rows = []
    source_scores = {}
    selected_records = []
    for condition, seed in jobs:
        candidates = [r for r in raw_records if r["condition"] == condition and r["seed"] == seed]
        acceptable = [r for r in candidates if valid(r)]
        if not acceptable:
            raise SystemExit(f"FAIL: no complete output for {condition}, seed {seed}; inspect {RAW.name}")
        selected_records.append(acceptable[-1])
    for record in selected_records:
        predictions, batch_error = parse_batch(record, pages)
        for page in pages:
            gold = {e["id"]: e["label"] for e in page["entities"]}
            predicted = predictions.get(page["page_id"], {})
            result = score(gold, predicted)
            row = {
                "page_id": page["page_id"],
                "condition": record["condition"],
                "selected_source": record["condition"],
                "seed": record["seed"],
                **{k: v for k, v in result.items() if k != "per_label_f1"},
                **{f"f1_{k}": v for k, v in result["per_label_f1"].items()},
                "parse_error": batch_error or "",
            }
            rows.append(row)
            source_scores[(page["page_id"], record["condition"], record["seed"])] = row

    # ponytail: entity count is a fixed, label-blind routing proxy; replace with measured parser cost on a larger benchmark.
    for page in pages:
        source = "layout_aware" if len(page["entities"]) >= 40 else "text_only"
        for seed in SEEDS:
            row = source_scores[(page["page_id"], source, seed)].copy()
            row.update(condition="hybrid", selected_source=source)
            rows.append(row)
    rows.sort(key=lambda r: (r["condition"], r["seed"], r["page_id"]))
    with PER_PAGE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stats = aggregate(rows, selected_records)
    dispositions = Counter(record_status(record, pages) for record in raw_records)
    created = sorted(
        int(record["response"]["created"])
        for record in raw_records
        if (record.get("response") or {}).get("created") is not None
    )
    result = {
        "experiment": "FUNSD real-document LLM entity classification",
        "artifact_generated_at": generated_at,
        "model_attempt_window_utc": {
            "first": datetime.fromtimestamp(created[0], timezone.utc).isoformat(),
            "last": datetime.fromtimestamp(created[-1], timezone.utc).isoformat(),
        },
        "endpoint": ENDPOINT,
        "model": MODEL,
        "sampling": {"seeds": SEEDS, "temperature": TEMPERATURE, "top_p": TOP_P},
        "dataset_sha256": sha256(DATASET),
        "sample_ids": SAMPLE_IDS,
        "sample_pages": len(pages),
        "evaluated_entities": sum(len(p["entities"]) for p in pages),
        "excluded_empty_text_entities": sum(p["excluded_empty"] for p in pages),
        "successful_model_calls": len(selected_records),
        "raw_model_attempts": len(raw_records),
        "excluded_pilot_or_invalid_attempts": len(raw_records) - len(selected_records),
        "attempt_disposition": dict(sorted(dispositions.items())),
        "selected_calls": [
            {
                "condition": record["condition"],
                "seed": record["seed"],
                "response_id": record["response"]["id"],
                "system_fingerprint": record["response"].get("system_fingerprint"),
            }
            for record in selected_records
        ],
        "ocr_boundary": "FUNSD-published OCR entity text and boxes; Tesseract was not installed and OCR was not rerun.",
        "hybrid_rule": "layout_aware when nonempty entity count >= 40; otherwise text_only; outputs selected post hoc from matched seeded baseline calls",
        "hybrid_cost_boundary": "No separate hybrid API call was made; hybrid prompt tokens are therefore not reported.",
        **stats,
    }
    AGGREGATE.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "dataset": "FUNSD original release",
        "official_url": "https://guillaumejaume.github.io/FUNSD/dataset.zip",
        "terms_url": "https://guillaumejaume.github.io/FUNSD/work/",
        "retrieved_date": "2026-07-20",
        "archive_sha256": sha256(DATASET),
        "archive_bytes": DATASET.stat().st_size,
        "split": "official testing_data split",
        "selection": "first four annotation filenames in lexicographic order, fixed before model execution",
        "selected_page_ids": SAMPLE_IDS,
        "pages": [
            {k: p[k] for k in ("page_id", "annotation_path", "image_path", "annotation_sha256", "image_sha256", "width", "height", "excluded_empty")}
            | {"evaluated_entities": len(p["entities"])}
            for p in pages
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failures = [r for r in rows if r["condition"] != "hybrid" and r["parse_error"]]
    if failures:
        raise SystemExit(f"FAIL: {len(failures)} page results have an unparseable batch output; inspect {RAW.name}")
    lines = [
        f"artifact_generated_at={generated_at}", "status=PASS", f"endpoint={ENDPOINT}", f"model={MODEL}",
        f"model_attempt_window_utc={result['model_attempt_window_utc']['first']}..{result['model_attempt_window_utc']['last']}",
        f"dataset_sha256={result['dataset_sha256']}", f"pages={result['sample_pages']}",
        f"entities={result['evaluated_entities']}", f"successful_model_calls={result['successful_model_calls']}",
        f"raw_model_attempts={result['raw_model_attempts']}",
        f"excluded_pilot_or_invalid_attempts={result['excluded_pilot_or_invalid_attempts']}",
        "attempt_disposition=" + json.dumps(result["attempt_disposition"], sort_keys=True),
    ]
    for summary in result["summary"]:
        lines.append(
            f"{summary['condition']}: accuracy={summary['accuracy_mean']:.4f}+/-{summary['accuracy_sd']:.4f}, "
            f"page_macro_f1={summary['mean_page_macro_f1_mean']:.4f}+/-{summary['mean_page_macro_f1_sd']:.4f}, "
            f"prompt_tokens={summary['prompt_tokens_mean']}"
        )
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("PASS: six checkpointed real-model calls; wrote raw, page-level, aggregate, manifest, and log artifacts")


if __name__ == "__main__":
    main()
