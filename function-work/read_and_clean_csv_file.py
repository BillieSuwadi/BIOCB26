from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


SUBJECT_ID_INDEX = 0
ARCHIVE_DATE_INDEX = 7
IMAGE_TYPE_INDEX = 22

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
)


def normalize_header(header: str) -> str:
    return header.strip().lower().replace("_", " ")


def find_column_index(headers: list[str], target_name: str, fallback_index: int) -> int:
    normalized_headers = [normalize_header(header) for header in headers]
    normalized_target = normalize_header(target_name)

    for index, header in enumerate(normalized_headers):
        if header == normalized_target:
            return index

    if fallback_index >= len(headers):
        raise ValueError(f"Required column '{target_name}' was not found and fallback index is out of range.")

    return fallback_index


def parse_archive_date(value: str) -> tuple[int, datetime | str]:
    cleaned = value.strip()
    if not cleaned:
        return 1, ""

    for fmt in DATE_FORMATS:
        try:
            return 0, datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    try:
        return 0, datetime.fromisoformat(cleaned)
    except ValueError:
        return 1, cleaned


def clean_csv_file(csv_path: Path) -> Path:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as source_file:
        reader = csv.reader(source_file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    headers = rows[0]
    data_rows = rows[1:]

    subject_id_idx = find_column_index(headers, "Subject_id", SUBJECT_ID_INDEX)
    archive_date_idx = find_column_index(headers, "Archive Date", ARCHIVE_DATE_INDEX)
    image_type_idx = find_column_index(headers, "Image Type", IMAGE_TYPE_INDEX)

    filtered_rows = []
    for row in data_rows:
        if not row:
            continue

        padded_row = row + [""] * max(0, len(headers) - len(row))
        image_type = padded_row[image_type_idx].strip().lower()
        if image_type == "mask":
            continue

        filtered_rows.append(padded_row[: len(headers)])

    filtered_rows.sort(
        key=lambda row: (
            row[subject_id_idx].strip(),
            parse_archive_date(row[archive_date_idx]),
        )
    )

    output_path = csv_path.with_name(f"{csv_path.stem}_cleaned.csv")
    with output_path.open("w", newline="", encoding="utf-8-sig") as target_file:
        writer = csv.writer(target_file)
        writer.writerow(headers)
        writer.writerows(filtered_rows)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a CSV file, remove rows whose Image Type is 'mask', "
            "sort by subject_id and Archive Date, and write a cleaned CSV."
        )
    )
    parser.add_argument("csv_file", help="Path to the CSV file to clean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_file).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {csv_path}")

    output_path = clean_csv_file(csv_path)
    print(f"Cleaned CSV written to: {output_path}")


if __name__ == "__main__":
    main()
