#!/usr/bin/env python3
import argparse
import os
import sys
import zipfile

def unzip_all(root_dir: str, delete_zip: bool) -> None:
    if not os.path.isdir(root_dir):
        raise ValueError(f"Not a directory: {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.lower().endswith(".zip"):
                continue
            zip_path = os.path.join(dirpath, name)
            out_dir = os.path.splitext(zip_path)[0]
            os.makedirs(out_dir, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(out_dir)
                print(f"Unzipped: {zip_path} -> {out_dir}")
                if delete_zip:
                    os.remove(zip_path)
                    print(f"Deleted: {zip_path}")
            except zipfile.BadZipFile:
                print(f"Bad zip file: {zip_path}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Unzip all .zip files under a directory.")
    parser.add_argument("path", help="Root directory to search for zip files")
    parser.add_argument("--delete", action="store_true",
                        help="Delete zip files after successful extraction")
    args = parser.parse_args()

    unzip_all(args.path, args.delete)

if __name__ == "__main__":
    main()