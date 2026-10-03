#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path


QUOTE_HEADER = [
    "quoteID",
    "quoteText",
    "subQuotationList",
    "quoteByteSpans",
    "speaker",
    "addressees",
    "quoteType",
    "referringExpression",
    "mentionTextsList",
    "mentionSpansList",
    "mentionEntitiesList",
]

CHAR_HEADER = [
    "Character ID",
    "Main Name",
    "Aliases",
    "Gender",
    "Category",
]


NON_CHARACTER_CUES = {
    "ACT",
    "ACT I",
    "ACT II",
    "ACT III",
    "ACT IV",
    "ACT V",
    "SCENE",
    "SCENE I",
    "SCENE II",
    "SCENE III",
    "SCENE IV",
    "SCENE V",
    "ENTER",
    "EXIT",
    "EXEUNT",
    "CURTAIN",
    "THE END",
}


def normalize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[:.]+$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.title()


def is_speaker_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    s_no_colon = re.sub(r"[:.]+$", "", s).strip()

    if s_no_colon.upper() in NON_CHARACTER_CUES:
        return False

    if len(s_no_colon) > 45:
        return False

    if re.search(r"[a-z]", s_no_colon):
        return False

    if not re.search(r"[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸÆŒ]", s_no_colon):
        return False

    if re.fullmatch(r"[IVXLCDM]+", s_no_colon):
        return False

    return True


def clean_dialogue(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_subquotes(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def parse_play_text(text: str) -> tuple[list[dict], dict]:
    lines = text.splitlines(keepends=True)

    rows = []
    characters = {}

    current_speaker = None
    current_chunks = []
    current_start_byte = None
    current_end_byte = None

    byte_pos = 0
    quote_num = 0

    def flush():
        nonlocal quote_num, current_speaker, current_chunks
        nonlocal current_start_byte, current_end_byte

        if not current_speaker or not current_chunks:
            return

        raw = "".join(current_chunks)
        quote_text = clean_dialogue(raw)

        if not quote_text:
            return

        speaker = normalize_name(current_speaker)
        characters.setdefault(speaker, {"Aliases": {speaker}})

        subquotes = split_subquotes(quote_text)

        rows.append(
            {
                "quoteID": f"Q{quote_num}",
                "quoteText": quote_text,
                "subQuotationList": json.dumps(subquotes, ensure_ascii=False),
                "quoteByteSpans": json.dumps([[current_start_byte, current_end_byte]], ensure_ascii=False),
                "speaker": speaker,
                "addressees": "[]",
                "quoteType": "Explicit",
                "referringExpression": "",
                "mentionTextsList": "[]",
                "mentionSpansList": "[]",
                "mentionEntitiesList": "[]",
            }
        )

        quote_num += 1

    for line in lines:
        stripped = line.strip()
        encoded_len = len(line.encode("utf-8"))

        if is_speaker_line(stripped):
            flush()
            current_speaker = stripped
            current_chunks = []
            current_start_byte = byte_pos + encoded_len
            current_end_byte = current_start_byte
        else:
            if current_speaker:
                current_chunks.append(line)
                current_end_byte = byte_pos + encoded_len

        byte_pos += encoded_len

    flush()

    return rows, characters


def write_quotation_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUOTE_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_character_csv(characters: dict, path: Path) -> None:
    names = sorted(characters.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHAR_HEADER)
        writer.writeheader()

        for idx, name in enumerate(names):
            writer.writerow(
                {
                    "Character ID": idx,
                    "Main Name": name,
                    "Aliases": "{" + repr(name) + "}",
                    "Gender": "",
                    "Category": "",
                }
            )


def process_file(txt_path: Path, output_dir: Path) -> None:
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    rows, characters = parse_play_text(text)

    base = txt_path.stem

    quotation_path = output_dir / f"{base}_quotation_info.csv"
    character_path = output_dir / f"{base}_character_info.csv"

    write_quotation_csv(rows, quotation_path)
    write_character_csv(characters, character_path)

    print(f"Processed {txt_path.name}")
    print(f"  Dialogue rows: {len(rows)}")
    print(f"  Characters: {len(characters)}")
    print(f"  Wrote: {quotation_path}")
    print(f"  Wrote: {character_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing OCR txt files")
    parser.add_argument("output_dir", help="Directory for generated CSV files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        raise SystemExit(f"No txt files found in {input_dir}")

    for txt_path in txt_files:
        process_file(txt_path, output_dir)


if __name__ == "__main__":
    main()