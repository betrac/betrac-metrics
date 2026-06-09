"""The two-stage SOAP judging pipeline.

``run_extract`` (stage 1) asks the LLM to extract atomic claims from each note;
``run_judge`` (stage 2) asks it to score each note against its transcript and
claims. Both run a thread pool, are resume-safe (a per-dialog raw cache lets a
re-run skip already-completed LLM calls), and never abort the whole batch on a
single failure — a failed dialog is recorded and skipped.

``aggregate_soap`` reduces per-dialog scores to means/stds, mirroring
``btc_eval.metrics`` aggregation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from btc_eval.soap_judge.backends import LLMBackend
from btc_eval.soap_judge.parse import parse_claims, parse_judgment
from btc_eval.soap_judge.prompts import load_prompt
from btc_eval.types import SOAP_SUBSCORES, SoapScores

log = logging.getLogger(__name__)

_EMPTY_CLAIMS = {"doc_type": "soap_claims", "claims": []}


def parse_shard_spec(spec: str) -> tuple[int, int]:
    """Parse a ``'i/N'`` shard spec into ``(index, count)``.

    Raises ValueError unless ``N >= 1`` and ``0 <= i < N``.
    """
    try:
        index_str, count_str = str(spec).split("/")
        index, count = int(index_str), int(count_str)
    except (ValueError, AttributeError):
        raise ValueError(f"--shard must be 'i/N' (e.g. 3/16), got {spec!r}") from None
    if count < 1 or not (0 <= index < count):
        raise ValueError(f"--shard out of range: {spec!r} (need 0 <= i < N, N >= 1)")
    return index, count


def select_shard(keys: Iterable[str], index: int, count: int) -> list[str]:
    """Return the subset of ``keys`` assigned to shard ``index`` of ``count``.

    Uses a stable SHA-1 hash of the key, so the partition is deterministic
    across processes and machines (the same dialog id always lands in the same
    shard). The shards are disjoint and their union is all keys.
    """
    return [
        k
        for k in keys
        if int(hashlib.sha1(str(k).encode("utf-8")).hexdigest(), 16) % count == index
    ]


def _load_cached_raw(raw_dir: Path | None, dialog_id: str) -> str | None:
    """Return the cached raw assistant text for a dialog, or None."""
    if raw_dir is None:
        return None
    path = raw_dir / f"{dialog_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    raw = data.get("raw_content")
    return raw if isinstance(raw, str) and raw.strip() else None


def _write_cached_raw(raw_dir: Path | None, dialog_id: str, model: str, raw: str) -> None:
    """Persist one raw assistant response (resume cache)."""
    if raw_dir is None:
        return
    path = raw_dir / f"{dialog_id}.json"
    payload = {"id": dialog_id, "model": model, "raw_content": raw}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_stage(
    items: Iterable[tuple[str, object]],
    work: Callable[[str, object], dict],
    workers: int,
    desc: str,
) -> dict[str, dict]:
    """Run ``work(id, payload)`` over items in a thread pool, keyed by id.

    Each ``work`` result may carry an ``elapsed_sec`` float and a ``cached``
    bool; this loop uses them to print a live progress line on stderr — count,
    last call duration, running average per LLM call, wall-clock, and an ETA —
    and logs a per-stage timing summary. Per-dialog timings are logged at DEBUG.
    """
    items = list(items)
    total = len(items)
    results: dict[str, dict] = {}
    if not items:
        return results

    stage_start = time.perf_counter()
    call_times: list[float] = []  # durations of actual (non-cached) LLM calls
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_id = {executor.submit(work, did, payload): did for did, payload in items}
        done = 0
        for future in as_completed(future_to_id):
            dialog_id = future_to_id[future]
            try:
                result = future.result()
            except Exception as exc:  # defensive — work() catches its own errors
                log.warning("[%s] %s failed: %s", desc, dialog_id, exc)
                result = {"status": "error", "error": str(exc)}
            results[dialog_id] = result
            done += 1

            elapsed = result.get("elapsed_sec")
            cached = result.get("cached", False)
            if elapsed is not None:
                log.debug("[%s] %s %s in %.1fs (status=%s)", desc, dialog_id,
                          "cached" if cached else "computed", elapsed,
                          result.get("status"))
                if not cached:
                    call_times.append(elapsed)

            wall = time.perf_counter() - stage_start
            if call_times:
                avg = sum(call_times) / len(call_times)
                # ETA: remaining items spread across the worker pool, at avg/call.
                eta = (total - done) * avg / max(1, workers)
                last = f"{elapsed:.0f}s" if elapsed is not None else "cached"
                print(
                    f"  {desc}: {done}/{total} | last {last} | avg {avg:.0f}s/call "
                    f"| wall {wall / 60:.1f}m | eta ~{eta / 60:.0f}m",
                    file=sys.stderr,
                )
            elif done % 10 == 0 or done == total:
                print(f"  {desc}: {done}/{total} | wall {wall / 60:.1f}m "
                      f"(resumed from cache)", file=sys.stderr)

    if call_times:
        ordered = sorted(call_times)
        median = ordered[len(ordered) // 2]
        log.info(
            "[%s] timing: %d LLM calls, mean %.1fs, median %.1fs, "
            "min %.1fs, max %.1fs, stage wall %.1fs",
            desc, len(call_times), sum(call_times) / len(call_times), median,
            ordered[0], ordered[-1], time.perf_counter() - stage_start,
        )
    return results


def run_extract(
    predictions: dict[str, str],
    backend: LLMBackend,
    model: str,
    *,
    workers: int = 10,
    max_tokens: int = 8000,
    raw_dir: str | Path | None = None,
    prompt_dir: str | None = None,
    resume: bool = True,
    system: str | None = None,
) -> dict[str, dict]:
    """Stage 1 — extract atomic claims from each SOAP note.

    Args:
        predictions: ``{dialog_id: soap_note_text}``.
        backend: any :class:`LLMBackend`.
        model: provider-specific model id.
        raw_dir: if given, raw responses are cached here for resume.
        resume: skip the LLM call when a cached raw response already parses.

    Returns ``{dialog_id: {"raw_content", "claims", "status"}}`` where ``claims``
    is the parsed object or ``None`` and ``status`` is ``parsed`` / ``parse_failed``
    / ``llm_error``.
    """
    template = load_prompt("extract_claims", prompt_dir)
    raw_path = Path(raw_dir) if raw_dir else None
    if raw_path:
        raw_path.mkdir(parents=True, exist_ok=True)

    def _work(dialog_id: str, note: object) -> dict:
        t0 = time.perf_counter()
        cached = _load_cached_raw(raw_path, dialog_id) if resume else None
        was_cached = cached is not None
        if was_cached:
            raw = cached
        else:
            prompt = template.format(soap_note=note)
            try:
                raw = backend.complete(
                    [{"role": "user", "content": prompt}],
                    model,
                    system=system,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                log.warning("[extract] LLM error for %s: %s", dialog_id, exc)
                return {"raw_content": "", "claims": None, "status": "llm_error",
                        "error": str(exc), "elapsed_sec": time.perf_counter() - t0,
                        "cached": False}
            _write_cached_raw(raw_path, dialog_id, model, raw)
        claims = parse_claims(raw)
        return {
            "raw_content": raw,
            "claims": claims,
            "status": "parsed" if claims is not None else "parse_failed",
            "elapsed_sec": time.perf_counter() - t0,
            "cached": was_cached,
        }

    return _run_stage(predictions.items(), _work, workers, "extract")


def run_judge(
    predictions: dict[str, str],
    transcripts: dict[str, str],
    claims_by_id: dict[str, dict],
    backend: LLMBackend,
    model: str,
    *,
    workers: int = 10,
    max_tokens: int = 8000,
    raw_dir: str | Path | None = None,
    prompt_dir: str | None = None,
    resume: bool = True,
    system: str | None = None,
) -> dict[str, dict]:
    """Stage 2 — judge each note against its transcript and extracted claims.

    Only dialogs present in both ``predictions`` and ``transcripts`` are judged.
    ``claims_by_id`` maps a dialog id to its stage-1 result dict (the ``claims``
    object is reused; a failed/missing extraction degrades to an empty claim
    list rather than skipping the dialog).

    Returns ``{dialog_id: {"raw_content", "judgment", "status"}}`` where
    ``judgment`` is the parsed judgment object or ``None``.
    """
    template = load_prompt("judge_soap_note", prompt_dir)
    raw_path = Path(raw_dir) if raw_dir else None
    if raw_path:
        raw_path.mkdir(parents=True, exist_ok=True)

    shared_ids = sorted(set(predictions) & set(transcripts))
    missing_transcripts = sorted(set(predictions) - set(transcripts))
    if missing_transcripts:
        print(
            f"Warning: {len(missing_transcripts)} predictions have no transcript "
            f"(skipped): {missing_transcripts[:5]}"
            f"{'...' if len(missing_transcripts) > 5 else ''}",
            file=sys.stderr,
        )

    def _work(dialog_id: str, _payload: object) -> dict:
        t0 = time.perf_counter()
        note = predictions[dialog_id]
        transcript = transcripts[dialog_id]
        extraction = (claims_by_id.get(dialog_id) or {}).get("claims") or _EMPTY_CLAIMS
        cached = _load_cached_raw(raw_path, dialog_id) if resume else None
        was_cached = cached is not None
        if was_cached:
            raw = cached
        else:
            prompt = template.format(
                transcript=transcript,
                soap_note=note,
                claims_json=json.dumps(extraction, ensure_ascii=False),
            )
            try:
                raw = backend.complete(
                    [{"role": "user", "content": prompt}],
                    model,
                    system=system,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                log.warning("[judge] LLM error for %s: %s", dialog_id, exc)
                return {"raw_content": "", "judgment": None, "status": "llm_error",
                        "error": str(exc), "elapsed_sec": time.perf_counter() - t0,
                        "cached": False}
            _write_cached_raw(raw_path, dialog_id, model, raw)
        judgment = parse_judgment(raw)
        return {
            "raw_content": raw,
            "judgment": judgment,
            "status": "parsed" if judgment is not None else "parse_failed",
            "elapsed_sec": time.perf_counter() - t0,
            "cached": was_cached,
        }

    payload_items = [(did, None) for did in shared_ids]
    return _run_stage(payload_items, _work, workers, "judge")


def aggregate_soap(scores: list[SoapScores]) -> dict:
    """Reduce per-dialog scores to means (subscores also get a population std)."""
    n = len(scores)
    if n == 0:
        return {"num_dialogs": 0}

    agg: dict = {"num_dialogs": n}
    for field in SOAP_SUBSCORES:
        values = [getattr(s, field) for s in scores]
        mean = sum(values) / n
        agg[f"mean_{field}"] = mean
        agg[f"std_{field}"] = (sum((v - mean) ** 2 for v in values) / n) ** 0.5

    count_fields = (
        "over_medicalization",
        "under_medicalization",
        "over_specific",
        "hallucination_rate",
        "contradiction_rate",
        "missed_claims",
        "critical_omissions",
        "redundancy_count",
    )
    for field in count_fields:
        values = [getattr(s, field) for s in scores]
        agg[f"mean_{field}"] = sum(values) / n
    return agg
