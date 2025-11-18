#!/usr/bin/env python3
"""
Fix Label Studio JSON image paths for local serving
"""

import json
import sys
from pathlib import Path
from urllib.parse import unquote

def fix_image_path(image_url):
    """
    Convert Label Studio path to working local path
    From: /data/local-files/?d=plantvillage/train/images/STLS_image%20%28942%29.jpg
    To: /data/upload/plantvillage/train/images/STLS_image (942).jpg
    """
    if "/data/local-files/?d=" in image_url:
        # Extract path after ?d=
        path_part = image_url.split("?d=")[1]
        # URL decode
        decoded = unquote(path_part)
        # Return with /data/upload/ prefix
        return f"/data/upload/{decoded}"
    return image_url


def fix_json_file(input_file, output_file):
    """Fix all image paths in Label Studio JSON"""

    print(f"Reading: {input_file}")
    with open(input_file, 'r') as f:
        tasks = json.load(f)

    print(f"Found {len(tasks)} tasks")

    fixed_count = 0
    for task in tasks:
        if "data" in task and "image" in task["data"]:
            original = task["data"]["image"]
            fixed = fix_image_path(original)

            if original != fixed:
                task["data"]["image"] = fixed
                fixed_count += 1

    print(f"Fixed {fixed_count} image paths")

    print(f"Writing: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=2)

    print("✅ Done!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_labelstudio_paths.py <input.json> <output.json>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    fix_json_file(input_file, output_file)
