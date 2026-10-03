#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."  # run from repo root

INPUT_DIR="data/plays_pdf"
OUTPUT_DIR="data/ocr_raw"
TMP_DIR="data/tmp_ocr_pages"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"

for pdf in "$INPUT_DIR"/*.pdf; do
    [ -e "$pdf" ] || continue

    filename=$(basename "$pdf")
    base="${filename%.pdf}"

    echo "========================================"
    echo "Processing: $filename"
    echo "========================================"

    workdir="$TMP_DIR/$base"
    mkdir -p "$workdir"

    echo "[1/3] Converting PDF pages to images..."
    pdftoppm -r 300 -png "$pdf" "$workdir/page" > /dev/null 2>&1

    echo "[2/3] Running OCR..."

    # naive language detection from filename
    lang="eng"

    if [[ "$filename" =~ [Ff]emme|[Ff]ran[cç]ais|[Bb]rieux ]]; then
        lang="fra"
    fi

    for img in "$workdir"/page-*.png; do
        [ -e "$img" ] || continue

        echo "OCR: $(basename "$img")"

        tesseract \
            "$img" \
            "${img%.png}" \
            -l "$lang" \
            --psm 6 \
            > /dev/null 2>&1
    done

    echo "[3/3] Combining text files..."

    cat "$workdir"/page-*.txt > "$OUTPUT_DIR/$base.txt"

    echo "Saved:"
    echo "$OUTPUT_DIR/$base.txt"

    rm -rf "$workdir"

    echo
done

echo "Done."