#!/usr/bin/env python3
"""Long-video multilingual transcription with resumable segments and translation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL = Path.home() / ".local/share/whisper.cpp/ggml-large-v3-turbo-q5_0.bin"
CACHE_SCHEMA = "translation-cache-v2"
MANIFEST_SCHEMA = "transcribe-video-long-manifest-v1"
DEFAULT_OUTPUT_ROOT = Path(".teaching-video-runtime") / "outputs" / "transcribe-video-long"


def safe_path_component(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "-", value.strip())
    safe = re.sub(r"-{2,}", "-", safe).strip(".-")
    return safe[:80] or "video"


def default_output_dir(video: Path, source_meta: dict[str, Any]) -> Path:
    digest = str(source_meta.get("sha256") or "")[:10] or "nohash"
    return Path.cwd() / DEFAULT_OUTPUT_ROOT / f"{safe_path_component(video.stem)}-{digest}"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def stamp(s: str) -> float:
    s = s.strip().replace(".", ",")
    h, m, rest = s.split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def fmt(t: float, comma: str = ",") -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{comma}{ms:03d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "path": str(path.expanduser().resolve()),
        "exists": path.is_file(),
    }
    if path.is_file():
        stat = path.stat()
        meta.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if hash_file:
            meta["sha256"] = sha256_file(path)
    return meta


def command_stdout(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()


def command_version(name: str) -> str | None:
    path = shutil.which(name)
    if not path:
        return None
    try:
        if name == "yt-dlp":
            return command_stdout([path, "--version"]).splitlines()[0]
        return command_stdout([path, "--version"]).splitlines()[0]
    except Exception:
        try:
            return command_stdout([path, "-version"]).splitlines()[0]
        except Exception:
            return "available"


def ffprobe_media(path: Path) -> dict[str, Any]:
    output = command_stdout([
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of", "json", str(path),
    ])
    return json.loads(output)


def parse_srt(path: Path) -> list[tuple[str, str, str]]:
    cues = []
    if not path.exists():
        return cues
    for block in re.split(r"\n\s*\n", path.read_text(errors="ignore").strip()):
        lines = block.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start, end = lines[1].split(" --> ", 1)
        cues.append((start, end, " ".join(x.strip() for x in lines[2:] if x.strip())))
    return cues


def parse_srt_blocks(path: Path) -> list[dict[str, Any]]:
    blocks = []
    if not path.exists():
        return blocks
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="ignore").strip()):
        lines = block.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start, end = lines[1].split(" --> ", 1)
        blocks.append({"start": stamp(start), "end": stamp(end), "text_lines": lines[2:]})
    return blocks


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def cache_key(text: str, fingerprint: str) -> str:
    payload = json.dumps(
        {"source": text, "translation_fingerprint": fingerprint},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def translation_fingerprint(
    language: str,
    model: Path,
    mode: str,
    window_cues: int,
    prompt_version: str = "zh-window-v2",
) -> str:
    model_meta = file_meta(model, hash_file=True)
    payload = {
        "language": language,
        "model_path": model_meta.get("path"),
        "model_sha256": model_meta.get("sha256"),
        "model_size": model_meta.get("size"),
        "prompt_version": prompt_version,
        "mode": mode,
        "window_cues": window_cues,
        "decoder": {
            "command": "llama-completion",
            "context": 2048,
            "tokens": 512,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.6,
            "repeat_penalty": 1.05,
            "mode": "conversation-single-turn-simple-io",
        },
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_translation_cache(path: Path, fingerprint: str) -> dict[str, Any]:
    fresh = {
        "schema": CACHE_SCHEMA,
        "translation_fingerprint": fingerprint,
        "entries": {},
    }
    if not path.exists():
        return fresh
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fresh
    if (
        isinstance(data, dict)
        and data.get("schema") == CACHE_SCHEMA
        and data.get("translation_fingerprint") == fingerprint
        and isinstance(data.get("entries"), dict)
    ):
        return data
    # Older cache files used raw source text as keys and did not include model/prompt identity.
    # Do not reuse them, because that can silently mix results from different translation settings.
    return fresh


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_llama_translation(prompt: str, model: Path) -> str:
    cmd = [
        "llama-completion", "-m", str(model), "-c", "2048", "-n", "512",
        "-cnv", "--single-turn", "--simple-io", "--no-display-prompt",
        "--temp", "0.7", "--top-k", "20", "--top-p", "0.6", "--repeat-penalty", "1.05",
        "-p", prompt,
    ]
    process = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=300)
    result = re.sub(r"\s*\[end of text\]\s*$", "", process.stdout).strip()
    if not result:
        raise RuntimeError("翻译模型返回空内容；请检查 llama-completion 推理日志。")
    return result


def translate_one(text: str, model: Path, cache: dict[str, Any], fingerprint: str) -> str:
    if not text.strip():
        return ""
    key = cache_key(text, fingerprint)
    entries = cache.setdefault("entries", {})
    hit = entries.get(key)
    if isinstance(hit, dict) and hit.get("translation"):
        return str(hit["translation"])
    prompt = "将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：\n" + text
    result = run_llama_translation(prompt, model)
    entries[key] = {"source": text, "translation": result}
    return result


def numbered_prompt(texts: list[str]) -> str:
    numbered = "\n".join(f"[{index:03d}] {text}" for index, text in enumerate(texts, 1))
    return (
        "你是专业字幕翻译。请把下面编号英文字幕翻译成自然、流畅的简体中文。\n"
        "要求：\n"
        "1. 利用相邻字幕作为上下文，修正被字幕切开的短语或句子。\n"
        "2. 必须保留每条字幕的编号，且输出条数与输入完全一致。\n"
        "3. 每行格式必须是 [001] 中文译文；不要输出解释、标题或空行。\n"
        "4. 人名、书名、品牌名不确定时保留原文或使用通行译法。\n\n"
        + numbered
    )


def parse_numbered_translations(text: str, expected: int) -> list[str] | None:
    found: dict[int, str] = {}
    current: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"^\[?(\d{1,3})\]?\s*[\.\)、:：-]?\s*(.+)$", line)
        if match:
            current = int(match.group(1))
            found[current] = match.group(2).strip()
        elif current is not None:
            found[current] = (found[current] + " " + line).strip()
    if all(index in found and found[index] for index in range(1, expected + 1)):
        translations = [found[index] for index in range(1, expected + 1)]
        if all(contains_cjk(item) for item in translations):
            return translations
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == expected:
        translations = [re.sub(r"^\[?\d{1,3}\]?\s*[\.\)、:：-]?\s*", "", line).strip() for line in lines]
        if all(contains_cjk(item) for item in translations):
            return translations
    return None


def ends_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(re.search(r'[.!?。！？]["”\')\]]?$', stripped))


def iter_translation_groups(
    cues: list[tuple[float, float, str]],
    max_cues: int,
    max_chars: int = 1200,
    min_cues_before_sentence_break: int = 3,
):
    group: list[tuple[float, float, str]] = []
    chars = 0
    for cue in cues:
        group.append(cue)
        chars += len(cue[2])
        if (
            len(group) >= max_cues
            or chars >= max_chars
            or (len(group) >= min_cues_before_sentence_break and ends_sentence(cue[2]))
        ):
            yield group
            group = []
            chars = 0
    if group:
        yield group


def build_sentence_groups(
    cues: list[tuple[float, float, str]],
    max_chars: int = 900,
) -> list[dict[str, Any]]:
    pieces: list[str] = []
    spans: list[tuple[int, int, float, float]] = []
    cursor = 0
    for start, end, text in cues:
        clean = text.strip()
        if not clean:
            continue
        if pieces:
            pieces.append(" ")
            cursor += 1
        begin = cursor
        pieces.append(clean)
        cursor += len(clean)
        spans.append((begin, cursor, start, end))
    full_text = "".join(pieces)
    if not full_text:
        return []

    def time_at(position: int, *, prefer_end: bool = False) -> float:
        for begin, end, cue_start, cue_end in spans:
            if begin <= position < end:
                if end <= begin:
                    return cue_end if prefer_end else cue_start
                ratio = (position - begin + (1 if prefer_end else 0)) / (end - begin)
                ratio = max(0.0, min(1.0, ratio))
                return cue_start + (cue_end - cue_start) * ratio
        return spans[-1][3] if prefer_end else spans[0][2]

    groups: list[dict[str, Any]] = []
    start_index = 0
    index = 0
    while index < len(full_text):
        char = full_text[index]
        should_split = char in ".!?。！？"
        # Handle a closing quote/bracket after the sentence-ending punctuation.
        split_end = index + 1
        closers = {'"', "'", "”", "’", ")", "]", "}"}
        while split_end < len(full_text) and full_text[split_end] in closers:
            split_end += 1
        too_long = split_end - start_index >= max_chars
        if should_split or too_long:
            text = full_text[start_index:split_end].strip()
            if text:
                groups.append({
                    "start": time_at(start_index),
                    "end": time_at(max(start_index, split_end - 1), prefer_end=True),
                    "text": text,
                })
            while split_end < len(full_text) and full_text[split_end].isspace():
                split_end += 1
            start_index = split_end
            index = split_end
            continue
        index += 1
    tail = full_text[start_index:].strip()
    if tail:
        groups.append({
            "start": time_at(start_index),
            "end": time_at(len(full_text) - 1, prefer_end=True),
            "text": tail,
        })
    return groups


def markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def translate_window(texts: list[str], model: Path, cache: dict[str, Any], fingerprint: str) -> list[str]:
    if not texts:
        return []
    if len(texts) == 1:
        return [translate_one(texts[0], model, cache, fingerprint)]
    payload = json.dumps({"sources": texts, "translation_fingerprint": fingerprint}, ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    entries = cache.setdefault("entries", {})
    hit = entries.get(key)
    if isinstance(hit, dict) and isinstance(hit.get("translations"), list) and len(hit["translations"]) == len(texts):
        return [str(item) for item in hit["translations"]]
    result = run_llama_translation(numbered_prompt(texts), model)
    translations = parse_numbered_translations(result, len(texts))
    if translations is None:
        # Keep the pipeline recoverable if the local model ignores numbering.
        translations = [translate_one(text, model, cache, fingerprint + ":fallback-cue") for text in texts]
    entries[key] = {"sources": texts, "translations": translations}
    return translations


def merge_srt(files: list[tuple[Path, float]], out: Path) -> None:
    blocks = []
    number = 1
    for path, offset in files:
        if not path.exists():
            continue
        for block in re.split(r"\n\s*\n", path.read_text(errors="ignore").strip()):
            lines = block.splitlines()
            if len(lines) < 3 or "-->" not in lines[1]:
                continue
            start, end = lines[1].split(" --> ")
            lines[0] = str(number)
            number += 1
            lines[1] = f"{fmt(stamp(start) + offset)} --> {fmt(stamp(end) + offset)}"
            blocks.append("\n".join(lines))
    out.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def segment_outputs_complete(prefix: Path) -> bool:
    return all(prefix.with_suffix(ext).is_file() and prefix.with_suffix(ext).stat().st_size > 0 for ext in (".txt", ".srt", ".vtt"))


def job_identity(source: dict[str, Any], language: str, segment_minutes: int, whisper_model: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_sha256": source.get("sha256"),
        "language": language,
        "segment_minutes": segment_minutes,
        "whisper_model_sha256": whisper_model.get("sha256"),
        "whisper_model_size": whisper_model.get("size"),
    }


def identity_matches(existing: dict[str, Any], current_identity: dict[str, Any]) -> bool:
    return existing.get("job_identity") == current_identity


def write_manifest(
    out: Path,
    args: argparse.Namespace,
    source_meta: dict[str, Any],
    media: dict[str, Any],
    whisper_meta: dict[str, Any],
    translation_meta: dict[str, Any] | None,
    translation_fp: str | None,
    segments: list[Path],
    outputs: dict[str, str],
    validation: dict[str, Any] | None = None,
) -> None:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "export_time": datetime.now(timezone.utc).isoformat(),
        "job_identity": job_identity(source_meta, args.language, args.segment_minutes, whisper_meta),
        "source": source_meta,
        "media": media,
        "parameters": {
            "language": args.language,
            "segment_minutes": args.segment_minutes,
            "translate_requested": args.translate,
            "translate_enabled": bool(translation_meta),
            "translation_mode": getattr(args, "translation_mode", None),
            "translation_window_cues": getattr(args, "translation_window_cues", None),
        },
        "tools": {
            "ffmpeg": command_version("ffmpeg"),
            "ffprobe": command_version("ffprobe"),
            "whisper-cli": command_version("whisper-cli"),
            "llama-completion": command_version("llama-completion"),
        },
        "models": {
            "whisper": whisper_meta,
            "translation": translation_meta,
            "translation_fingerprint": translation_fp,
        },
        "segments": [str(path) for path in segments],
        "outputs": outputs,
    }
    if validation is not None:
        manifest["validation"] = validation
    atomic_write_json(out / "manifest.json", manifest)


def validate_outputs(
    output_dir: Path,
    stem: str,
    expect_bilingual: bool = False,
    *,
    expect_cue_txt: bool = False,
    expect_vtt: bool = False,
    expect_sentences_txt: bool = False,
    expect_sentences_json: bool = False,
    emit_validation_json: bool = False,
) -> dict[str, Any]:
    base = output_dir / stem
    txt = base.with_suffix(".txt")
    srt = base.with_suffix(".srt")
    vtt = base.with_suffix(".vtt")
    errors: list[str] = []
    required = [srt]
    if expect_cue_txt:
        required.append(txt)
    if expect_vtt:
        required.append(vtt)
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing_or_empty:{path.name}")
    srt_blocks = parse_srt_blocks(srt)
    if not srt_blocks:
        errors.append("srt_has_no_cues")
    last_end = -1.0
    for index, block in enumerate(srt_blocks, 1):
        if block["start"] < last_end:
            errors.append(f"srt_not_monotonic:{index}")
        if block["end"] <= block["start"]:
            errors.append(f"srt_non_positive_duration:{index}")
        last_end = block["end"]
        if expect_bilingual:
            lines = [line.strip() for line in block["text_lines"] if line.strip()]
            if len(lines) < 2:
                errors.append(f"cue_not_bilingual:{index}")
            elif not contains_cjk("\n".join(lines[1:])):
                errors.append(f"cue_translation_not_chinese:{index}")
    if vtt.exists():
        content = vtt.read_text(encoding="utf-8", errors="ignore")
        if not content.startswith("WEBVTT"):
            errors.append("vtt_missing_header")
        vtt_times = re.findall(r"(\d\d:\d\d:\d\d\.\d\d\d) --> (\d\d:\d\d:\d\d\.\d\d\d)", content)
        last_end = -1.0
        for index, (start, end) in enumerate(vtt_times, 1):
            start_f = stamp(start)
            end_f = stamp(end)
            if start_f < last_end:
                errors.append(f"vtt_not_monotonic:{index}")
            if end_f <= start_f:
                errors.append(f"vtt_non_positive_duration:{index}")
            last_end = end_f
        if len(vtt_times) != len(srt_blocks):
            errors.append(f"vtt_srt_cue_count_mismatch:{len(vtt_times)}!={len(srt_blocks)}")
    cache_path = output_dir / ".segments" / "translations.json"
    cache_entries = None
    if expect_bilingual:
        if not cache_path.exists():
            errors.append("translation_cache_missing")
        else:
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                cache_entries = len(cache.get("entries", {})) if isinstance(cache.get("entries"), dict) else None
                if cache.get("schema") != CACHE_SCHEMA or cache_entries is None:
                    errors.append("translation_cache_schema_invalid")
            except Exception as exc:
                errors.append(f"translation_cache_unreadable:{exc}")
        sentence_md = output_dir / f"{stem}.sentences.md"
        sentence_required = [sentence_md]
        if expect_sentences_txt:
            sentence_required.append(output_dir / f"{stem}.sentences.txt")
        if expect_sentences_json:
            sentence_required.append(output_dir / f"{stem}.sentences.json")
        for path in sentence_required:
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing_or_empty:{path.name}")
    result = {
        "ok": not errors,
        "errors": errors,
        "stem": stem,
        "files": {
            "txt": str(txt) if txt.exists() else None,
            "srt": str(srt),
            "vtt": str(vtt) if vtt.exists() else None,
            "translation_cache": str(cache_path) if cache_path.exists() else None,
            "sentences_txt": str(output_dir / f"{stem}.sentences.txt") if (output_dir / f"{stem}.sentences.txt").exists() else None,
            "sentences_md": str(output_dir / f"{stem}.sentences.md") if (output_dir / f"{stem}.sentences.md").exists() else None,
            "sentences_json": str(output_dir / f"{stem}.sentences.json") if (output_dir / f"{stem}.sentences.json").exists() else None,
        },
        "srt_cues": len(srt_blocks),
        "translation_cache_entries": cache_entries,
    }
    if emit_validation_json:
        atomic_write_json(output_dir / "validation.json", result)
    if errors:
        raise SystemExit("validation failed: " + "; ".join(errors))
    return result


def validate_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Validate transcribe-video-long outputs")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stem", help="Output basename; defaults to the only .srt file in output_dir")
    parser.add_argument("--expect-bilingual", action="store_true")
    parser.add_argument("--expect-cue-txt", action="store_true")
    parser.add_argument("--expect-vtt", action="store_true")
    parser.add_argument("--expect-sentences-txt", action="store_true")
    parser.add_argument("--expect-sentences-json", action="store_true")
    parser.add_argument("--emit-validation-json", action="store_true")
    args = parser.parse_args(argv)
    args.video = args.video.expanduser()
    args.whisper_model = args.whisper_model.expanduser()
    if args.translation_model is not None:
        args.translation_model = args.translation_model.expanduser()
    stem = args.stem
    if not stem:
        candidates = [p.stem for p in args.output_dir.glob("*.srt")]
        if len(candidates) != 1:
            raise SystemExit("请用 --stem 指定输出 basename")
        stem = candidates[0]
    result = validate_outputs(
        args.output_dir,
        stem,
        args.expect_bilingual,
        expect_cue_txt=args.expect_cue_txt,
        expect_vtt=args.expect_vtt,
        expect_sentences_txt=args.expect_sentences_txt,
        expect_sentences_json=args.expect_sentences_json,
        emit_validation_json=args.emit_validation_json,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def transcribe_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, help="输出目录；默认写入当前仓库 .teaching-video-runtime/outputs/transcribe-video-long/<视频名>-<源文件hash前缀>/")
    parser.add_argument("-l", "--language", default="auto", help="zh/en/ja/... or auto")
    parser.add_argument("--whisper-model", type=Path, default=MODEL, help="whisper.cpp GGML/GGUF 模型路径；默认 ~/.local/share/whisper.cpp/ggml-large-v3-turbo-q5_0.bin")
    parser.add_argument("--segment-minutes", type=int, default=15)
    parser.add_argument("--translate", action="store_true", help="逐字幕翻译成简体中文并合并到同一输出")
    parser.add_argument("--translation-model", type=Path, help="llama-completion 可加载的 GGUF 翻译模型路径")
    parser.add_argument("--translation-mode", choices=("window", "cue"), default="window", help="window=按相邻字幕窗口翻译以利用上下文；cue=逐字幕独立翻译")
    parser.add_argument("--translation-window-cues", type=int, default=12, help="window 模式下每次给翻译模型的最大字幕条数")
    parser.add_argument("--full", action="store_true", help="产出所有调试/兼容文件，并保留音频缓存")
    parser.add_argument("--emit-cue-txt", action="store_true", help="额外产出 cue 级 TXT")
    parser.add_argument("--emit-vtt", action="store_true", help="额外产出 VTT 字幕")
    parser.add_argument("--emit-sentences-txt", action="store_true", help="额外产出逐句 TXT")
    parser.add_argument("--emit-sentences-json", action="store_true", help="额外产出逐句 JSON")
    parser.add_argument("--emit-validation-json", action="store_true", help="额外写出独立 validation.json；默认只写入 manifest.json")
    parser.add_argument("--keep-audio-cache", action="store_true", help="保留 .segments/audio.wav 和 .segments/audio/*.wav")
    parser.add_argument("--no-validate", action="store_true", help="跳过输出完整性校验")
    args = parser.parse_args(argv)
    args.video = args.video.expanduser()
    args.whisper_model = args.whisper_model.expanduser()
    if args.translation_model is not None:
        args.translation_model = args.translation_model.expanduser()
    if args.full:
        args.emit_cue_txt = True
        args.emit_vtt = True
        args.emit_sentences_txt = True
        args.emit_sentences_json = True
        args.emit_validation_json = True
        args.keep_audio_cache = True
    if args.segment_minutes <= 0:
        raise SystemExit("--segment-minutes 必须为正整数")
    if args.translation_window_cues <= 0:
        raise SystemExit("--translation-window-cues 必须为正整数")
    if not args.video.is_file():
        raise SystemExit(f"视频不存在: {args.video}")
    if not args.whisper_model.is_file():
        raise SystemExit(f"Whisper 模型不存在: {args.whisper_model}")

    source_meta = file_meta(args.video, hash_file=True)
    out = (args.output_dir or default_output_dir(args.video, source_meta)).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / ".segments"
    work.mkdir(exist_ok=True)

    whisper_meta = file_meta(args.whisper_model, hash_file=True)
    media = ffprobe_media(args.video)
    current_identity = job_identity(source_meta, args.language, args.segment_minutes, whisper_meta)
    existing_manifest_path = out / "manifest.json"
    if existing_manifest_path.exists() and any(work.glob("part_*.txt")):
        try:
            existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = {}
        if not identity_matches(existing_manifest, current_identity):
            raise SystemExit("输出目录已有不同源视频/语言/模型/切片参数的结果；请换一个 --output-dir，避免误复用旧转写。")

    translate_enabled = args.translate and args.language.lower() not in ("zh", "zh-cn", "cmn")
    if translate_enabled and not args.translation_model:
        raise SystemExit("--translate 需要 --translation-model（GGUF），例如 Hunyuan-MT Q4_K_M")
    if translate_enabled and not args.translation_model.is_file():
        raise SystemExit(f"翻译模型不存在: {args.translation_model}")

    translation_meta = file_meta(args.translation_model, hash_file=True) if translate_enabled else None
    translation_fp = translation_fingerprint(args.language, args.translation_model, args.translation_mode, args.translation_window_cues) if translate_enabled else None

    wav = work / "audio.wav"
    run([
        "ffmpeg", "-y", "-i", str(args.video), "-vn", "-ac", "1", "-ar", "16000",
        "-af", "highpass=f=80,lowpass=f=7600,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "pcm_s16le", str(wav),
    ])

    segdir = work / "audio"
    segdir.mkdir(exist_ok=True)
    run([
        "ffmpeg", "-y", "-i", str(wav), "-f", "segment",
        "-segment_time", str(args.segment_minutes * 60), "-c", "copy",
        str(segdir / "part_%04d.wav"),
    ])

    txt: list[Path] = []
    srts: list[tuple[Path, float]] = []
    vtts: list[tuple[Path, float]] = []
    segments = sorted(segdir.glob("part_*.wav"))
    for index, segment in enumerate(segments):
        prefix = work / f"part_{index:04d}"
        if not segment_outputs_complete(prefix):
            cmd = [
                "whisper-cli", "-m", str(args.whisper_model), "-f", str(segment), "-l", args.language,
                "-otxt", "-osrt", "-ovtt", "-of", str(prefix),
                "--no-fallback", "--no-prints",
            ]
            run(cmd)
        offset = index * args.segment_minutes * 60
        txt.append(prefix.with_suffix(".txt"))
        srts.append((prefix.with_suffix(".srt"), offset))
        vtts.append((prefix.with_suffix(".vtt"), offset))

    base = out / args.video.stem
    if translate_enabled:
        assert args.translation_model is not None
        assert translation_fp is not None
        cache_path = work / "translations.json"
        cache = load_translation_cache(cache_path, translation_fp)
        all_cues: list[tuple[float, float, str]] = []
        for srt_path, offset in srts:
            for start, end, text in parse_srt(srt_path):
                all_cues.append((stamp(start) + offset, stamp(end) + offset, text))
        bilingual: list[tuple[float, float, str]] = []
        if args.translation_mode == "cue":
            for start, end, text in all_cues:
                zh = translate_one(text, args.translation_model, cache, translation_fp)
                atomic_write_json(cache_path, cache)
                bilingual.append((start, end, text + "\n" + zh if zh else text))
        else:
            for group in iter_translation_groups(all_cues, args.translation_window_cues):
                translations = translate_window([item[2] for item in group], args.translation_model, cache, translation_fp)
                atomic_write_json(cache_path, cache)
                for (start, end, text), zh in zip(group, translations):
                    bilingual.append((start, end, text + "\n" + zh if zh else text))
        atomic_write_json(cache_path, cache)
        with base.with_suffix(".srt").open("w", encoding="utf-8") as target:
            for index, (start, end, text) in enumerate(bilingual, 1):
                target.write(f"{index}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n")
        if args.emit_cue_txt:
            base.with_suffix(".txt").write_text("\n\n".join(item[2] for item in bilingual) + "\n", encoding="utf-8")
        sentence_groups = build_sentence_groups(all_cues)
        sentence_items: list[dict[str, Any]] = []
        sentence_fp = translation_fp + ":sentences-v2"
        for item in sentence_groups:
            zh = translate_one(item["text"], args.translation_model, cache, sentence_fp)
            atomic_write_json(cache_path, cache)
            sentence_items.append({**item, "translation": zh})
        atomic_write_json(cache_path, cache)
        sentence_txt_parts = []
        sentence_md_rows = []
        for index, item in enumerate(sentence_items, 1):
            time_range = f"{fmt(item['start'], '.')} --> {fmt(item['end'], '.')}"
            sentence_txt_parts.append(
                f"{index}. [{time_range}]\n原文：{item['text']}\n中文：{item['translation']}"
            )
            sentence_md_rows.append(
                f"| {index} | {time_range} | {markdown_cell(item['text'])} | {markdown_cell(item['translation'])} |"
            )
        (out / f"{args.video.stem}.sentences.md").write_text(
            "# 逐句双语稿\n\n| # | 时间 | 原文 | 中文 |\n|---:|---|---|---|\n" + "\n".join(sentence_md_rows) + "\n",
            encoding="utf-8",
        )
        if args.emit_sentences_txt:
            (out / f"{args.video.stem}.sentences.txt").write_text("\n\n".join(sentence_txt_parts) + "\n", encoding="utf-8")
        if args.emit_sentences_json:
            (out / f"{args.video.stem}.sentences.json").write_text(
                json.dumps({"schema": "sentence-bilingual-v1", "items": sentence_items}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    else:
        if args.emit_cue_txt:
            base.with_suffix(".txt").write_text(
                "\n\n".join(path.read_text(errors="ignore").strip() for path in txt if path.exists()) + "\n",
                encoding="utf-8",
            )
        merge_srt(srts, base.with_suffix(".srt"))

    vtt = base.with_suffix(".vtt")
    if args.emit_vtt:
        vtt.write_text("WEBVTT\n\n", encoding="utf-8")
        if translate_enabled:
            with vtt.open("a", encoding="utf-8") as target:
                for start, end, text in bilingual:
                    target.write(f"{fmt(start, '.')} --> {fmt(end, '.')}\n{text}\n\n")
        else:
            with vtt.open("a", encoding="utf-8") as target:
                for vtt_path, offset in vtts:
                    if vtt_path.exists():
                        body = vtt_path.read_text(errors="ignore").replace("WEBVTT", "").strip()
                        for line in body.splitlines():
                            if " --> " in line:
                                start, end = line.split(" --> ")
                                line = f"{fmt(stamp(start) + offset, '.')} --> {fmt(stamp(end) + offset, '.')}"
                            target.write(line + "\n")
                        target.write("\n")

    outputs = {
        "txt": str(base.with_suffix(".txt")) if args.emit_cue_txt else None,
        "srt": str(base.with_suffix(".srt")),
        "vtt": str(base.with_suffix(".vtt")) if args.emit_vtt else None,
        "sentences_txt": str(out / f"{args.video.stem}.sentences.txt") if (translate_enabled and args.emit_sentences_txt) else None,
        "sentences_md": str(out / f"{args.video.stem}.sentences.md") if translate_enabled else None,
        "sentences_json": str(out / f"{args.video.stem}.sentences.json") if (translate_enabled and args.emit_sentences_json) else None,
        "validation": str(out / "validation.json") if args.emit_validation_json else None,
        "manifest": str(out / "manifest.json"),
    }
    for optional in [
        base.with_suffix(".txt"),
        base.with_suffix(".vtt"),
        out / f"{args.video.stem}.sentences.txt",
        out / f"{args.video.stem}.sentences.json",
        out / "validation.json",
    ]:
        keep = (
            (optional == base.with_suffix(".txt") and args.emit_cue_txt)
            or (optional == base.with_suffix(".vtt") and args.emit_vtt)
            or (optional.name.endswith(".sentences.txt") and args.emit_sentences_txt)
            or (optional.name.endswith(".sentences.json") and args.emit_sentences_json)
            or (optional.name == "validation.json" and args.emit_validation_json)
        )
        if optional.exists() and not keep:
            optional.unlink()
    validation = None
    if not args.no_validate:
        validation = validate_outputs(
            out,
            args.video.stem,
            expect_bilingual=translate_enabled,
            expect_cue_txt=args.emit_cue_txt,
            expect_vtt=args.emit_vtt,
            expect_sentences_txt=args.emit_sentences_txt,
            expect_sentences_json=args.emit_sentences_json,
            emit_validation_json=args.emit_validation_json,
        )
    write_manifest(out, args, source_meta, media, whisper_meta, translation_meta, translation_fp, segments, outputs, validation)
    if not args.keep_audio_cache:
        if wav.exists():
            wav.unlink()
        for audio_part in segdir.glob("part_*.wav"):
            audio_part.unlink()
    visible_outputs = [str(base.with_suffix(".srt"))]
    if translate_enabled:
        visible_outputs.insert(0, str(out / f"{args.video.stem}.sentences.md"))
    if args.emit_cue_txt:
        visible_outputs.append(str(base.with_suffix(".txt")))
    if args.emit_vtt:
        visible_outputs.append(str(base.with_suffix(".vtt")))
    print("输出目录: " + str(out))
    print("输出: " + " / ".join(visible_outputs))
    if validation:
        print("校验: ok")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate_cli(sys.argv[2:])
    else:
        transcribe_cli(sys.argv[1:])


if __name__ == "__main__":
    main()
