#!/usr/bin/env python3

"""Generate a multilingual OmniVoice quality-reference set via the API."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MANIFEST = Path("bench_assets/quality_refs/reference_requests.json")
DEFAULT_OUTPUT_DIR = Path("bench_assets/quality_refs/generated")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate multilingual OmniVoice quality-reference WAVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8002",
        help="Base URL for the running OmniVoice API server.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSON manifest of multilingual reference requests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store generated WAVs and headers.",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=16,
        help="Iterative decode steps for the reference set.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="Guidance scale for the reference set.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Fixed output duration in seconds.",
    )
    parser.add_argument(
        "--curl-bin",
        default="curl",
        help="curl binary to use.",
    )
    return parser


def _load_manifest(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("manifest must be a non-empty JSON array")
    required = {"id", "language", "instruct", "text"}
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {index} must be an object")
        missing = required.difference(item)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"manifest item {index} is missing: {missing_str}")
    return data


def _parse_http_status(headers_path: Path) -> int | None:
    raw = headers_path.read_text(encoding="utf-8", errors="replace")
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def _run_one(
    base_url: str,
    item: dict[str, str],
    output_dir: Path,
    num_step: int,
    guidance_scale: float,
    duration: float,
    curl_bin: str,
) -> dict[str, object]:
    item_id = item["id"]
    wav_path = output_dir / f"{item_id}.wav"
    headers_path = output_dir / f"{item_id}.headers.txt"
    text_path = output_dir / f"{item_id}.txt"

    cmd = [
        curl_bin,
        "-sS",
        "--fail-with-body",
        "-D",
        str(headers_path),
        "-o",
        str(wav_path),
        "-X",
        "POST",
        f"{base_url.rstrip('/')}/generate",
        "-F",
        "mode=design",
        "-F",
        f"language={item['language']}",
        "-F",
        f"text={item['text']}",
        "-F",
        f"instruct={item['instruct']}",
        "-F",
        f"num_step={num_step}",
        "-F",
        f"guidance_scale={guidance_scale}",
        "-F",
        f"duration={duration}",
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        if headers_path.exists():
            headers_path.unlink()
        if wav_path.exists():
            wav_path.unlink()
        stderr = completed.stderr.strip() or completed.stdout.strip() or "curl failed"
        raise RuntimeError(f"{item_id}: {stderr}")

    text_path.write_text(item["text"] + "\n", encoding="utf-8")
    http_status = _parse_http_status(headers_path)
    return {
        "id": item_id,
        "language": item["language"],
        "instruct": item["instruct"],
        "text": item["text"],
        "wav_path": str(wav_path),
        "headers_path": str(headers_path),
        "text_path": str(text_path),
        "http_status": http_status,
    }


def main() -> int:
    args = build_parser().parse_args()
    manifest = _load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for item in manifest:
        result = _run_one(
            base_url=args.base_url,
            item=item,
            output_dir=args.output_dir,
            num_step=args.num_step,
            guidance_scale=args.guidance_scale,
            duration=args.duration,
            curl_bin=args.curl_bin,
        )
        results.append(result)
        print(
            f"[ok] {result['id']} {result['language']} -> {result['wav_path']}"
        )

    index = {
        "base_url": args.base_url,
        "num_step": args.num_step,
        "guidance_scale": args.guidance_scale,
        "duration": args.duration,
        "items": results,
    }
    index_path = args.output_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {index_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
