#!/usr/bin/env python3
"""
Strawberry dataset cleaning pipeline.

Automatically filters out non-strawberry images from datasets using:
1. CLIP zero-shot classification (primary filter)
2. Fastdup outlier detection (optional secondary filter)

Usage:
    # Basic usage with CLIP filtering
    python scripts/clean_dataset.py \
        --input data/raw/merged \
        --output data/cleaned \
        --threshold 0.7

    # With Fastdup outlier detection
    python scripts/clean_dataset.py \
        --input data/raw/merged \
        --output data/cleaned \
        --threshold 0.7 \
        --use-fastdup

    # Dry run (report only, no file operations)
    python scripts/clean_dataset.py \
        --input data/raw/merged \
        --threshold 0.7 \
        --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")

    try:
        from transformers import pipeline  # noqa: F401
    except ImportError:
        missing.append("transformers")

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("pillow")

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


class StrawberryDatasetCleaner:
    """Clean strawberry datasets by filtering non-strawberry images."""

    # Classes that indicate strawberry content
    STRAWBERRY_CLASSES = [
        "strawberry fruit",
        "strawberry plant",
        "strawberry leaf",
        "strawberry flower",
        "ripe strawberry",
        "unripe strawberry",
        "strawberry disease",
        "diseased strawberry leaf",
    ]

    # Classes that indicate non-strawberry content
    OTHER_CLASSES = [
        "tomato",
        "tomato plant",
        "apple",
        "raspberry",
        "blackberry",
        "grape",
        "cherry",
        "blueberry",
        "other fruit",
        "other plant",
        "other leaf",
        "background",
        "text or label",
        "drawing or illustration",
    ]

    def __init__(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        threshold: float = 0.7,
        dry_run: bool = False,
        use_fastdup: bool = False,
        fastdup_threshold: float = 0.05,
    ):
        """
        Initialize the cleaner.

        Args:
            input_dir: Directory containing images to clean
            output_dir: Directory to save cleaned images (None for dry run)
            threshold: Minimum strawberry confidence to keep image (0-1)
            dry_run: If True, only report without copying files
            use_fastdup: If True, run fastdup outlier detection after CLIP
            fastdup_threshold: Outlier threshold for fastdup (lower = stricter)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.threshold = threshold
        self.dry_run = dry_run
        self.use_fastdup = use_fastdup
        self.fastdup_threshold = fastdup_threshold

        self.clip_pipeline = None
        self.stats = {
            "total": 0,
            "clip_passed": 0,
            "clip_failed": 0,
            "fastdup_outliers": 0,
            "final": 0,
            "errors": 0,
        }
        self.results = {}

    def _load_clip(self):
        """Load CLIP model lazily."""
        if self.clip_pipeline is None:
            from transformers import pipeline
            print("Loading CLIP model (openai/clip-vit-large-patch14)...")
            self.clip_pipeline = pipeline(
                "zero-shot-image-classification",
                model="openai/clip-vit-large-patch14",
                use_fast=True,
            )
            print("✓ CLIP model loaded")

    def _get_image_paths(self) -> list[Path]:
        """Get all image paths from input directory."""
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = []
        for ext in extensions:
            paths.extend(self.input_dir.glob(f"**/*{ext}"))
            paths.extend(self.input_dir.glob(f"**/*{ext.upper()}"))
        return sorted(set(paths))

    def _classify_image(self, image_path: Path) -> dict:
        """
        Classify a single image using CLIP.

        Returns:
            dict with strawberry_confidence, predictions, keep flag
        """
        from PIL import Image

        try:
            image = Image.open(image_path).convert("RGB")

            all_classes = self.STRAWBERRY_CLASSES + self.OTHER_CLASSES
            predictions = self.clip_pipeline(image, all_classes)

            # Calculate strawberry confidence
            strawberry_score = sum(
                p["score"] for p in predictions
                if any(sc in p["label"] for sc in self.STRAWBERRY_CLASSES)
            )

            # Get top predictions for reporting
            top_predictions = [
                {"label": p["label"], "score": round(p["score"], 4)}
                for p in predictions[:5]
            ]

            return {
                "strawberry_confidence": round(strawberry_score, 4),
                "top_predictions": top_predictions,
                "keep": strawberry_score >= self.threshold,
                "error": None,
            }

        except Exception as e:
            return {
                "strawberry_confidence": 0.0,
                "top_predictions": [],
                "keep": False,
                "error": str(e),
            }

    def run_clip_filter(self) -> dict[str, dict]:
        """
        Run CLIP zero-shot classification on all images.

        Returns:
            dict mapping image paths to classification results
        """
        self._load_clip()

        image_paths = self._get_image_paths()
        self.stats["total"] = len(image_paths)

        print(f"\nProcessing {len(image_paths)} images with CLIP...")
        print(f"Threshold: {self.threshold}")
        print("-" * 50)

        results = {}
        for i, img_path in enumerate(image_paths, 1):
            rel_path = str(img_path.relative_to(self.input_dir))
            result = self._classify_image(img_path)
            results[rel_path] = result

            if result["error"]:
                self.stats["errors"] += 1
                status = "ERROR"
            elif result["keep"]:
                self.stats["clip_passed"] += 1
                status = "KEEP"
            else:
                self.stats["clip_failed"] += 1
                status = "REMOVE"

            # Progress update
            if i % 50 == 0 or i == len(image_paths):
                pct = 100 * i / len(image_paths)
                print(f"[{i}/{len(image_paths)}] {pct:.1f}% - {status}: {rel_path[:50]}...")

        return results

    def run_fastdup_filter(self, passed_images: list[Path]) -> set[str]:
        """
        Run fastdup outlier detection on images that passed CLIP filter.

        Args:
            passed_images: List of image paths that passed CLIP filter

        Returns:
            Set of relative paths that are outliers
        """
        try:
            import fastdup
        except ImportError:
            print("⚠️  Fastdup not installed. Skipping outlier detection.")
            print("Install with: pip install fastdup")
            return set()

        if not passed_images:
            return set()

        print(f"\nRunning Fastdup outlier detection on {len(passed_images)} images...")

        # Create temp directory with passed images
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Symlink or copy images
            for img_path in passed_images:
                dst = temp_path / img_path.name
                if not dst.exists():
                    shutil.copy2(img_path, dst)

            # Run fastdup
            work_dir = temp_path / "fastdup_work"
            fd = fastdup.create(input_dir=str(temp_path), work_dir=str(work_dir))
            fd.run()

            # Get outliers
            outliers_df = fd.outliers()
            if outliers_df is not None and len(outliers_df) > 0:
                # Filter by threshold
                outlier_files = set(
                    outliers_df[outliers_df["outlier_score"] < self.fastdup_threshold]["filename"].tolist()
                )
                self.stats["fastdup_outliers"] = len(outlier_files)
                return outlier_files

        return set()

    def run(self) -> dict:
        """
        Run the full cleaning pipeline.

        Returns:
            dict with stats and detailed results
        """
        print("=" * 60)
        print("Strawberry Dataset Cleaner")
        print("=" * 60)
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir or '(dry run)'}")
        print(f"Threshold: {self.threshold}")
        print(f"Fastdup: {'enabled' if self.use_fastdup else 'disabled'}")

        # Step 1: CLIP filtering
        self.results = self.run_clip_filter()

        # Get paths of images that passed CLIP
        passed_images = [
            self.input_dir / rel_path
            for rel_path, result in self.results.items()
            if result["keep"]
        ]

        # Step 2: Fastdup outlier detection (optional)
        fastdup_outliers = set()
        if self.use_fastdup and passed_images:
            fastdup_outliers = self.run_fastdup_filter(passed_images)

            # Update results with fastdup findings
            for rel_path in self.results:
                if Path(rel_path).name in fastdup_outliers:
                    self.results[rel_path]["fastdup_outlier"] = True
                    self.results[rel_path]["keep"] = False

        # Step 3: Copy kept images (if not dry run)
        if not self.dry_run and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nCopying images to {self.output_dir}...")
            for rel_path, result in self.results.items():
                if result["keep"]:
                    src = self.input_dir / rel_path
                    dst = self.output_dir / rel_path
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

                    # Copy corresponding label file if exists
                    label_src = src.with_suffix(".txt")
                    if label_src.exists():
                        label_dst = dst.with_suffix(".txt")
                        shutil.copy2(label_src, label_dst)

                    self.stats["final"] += 1
        else:
            self.stats["final"] = sum(1 for r in self.results.values() if r["keep"])

        # Save report
        report = {
            "config": {
                "input_dir": str(self.input_dir),
                "output_dir": str(self.output_dir) if self.output_dir else None,
                "threshold": self.threshold,
                "use_fastdup": self.use_fastdup,
                "fastdup_threshold": self.fastdup_threshold,
                "dry_run": self.dry_run,
            },
            "stats": self.stats,
            "details": self.results,
        }

        report_path = (self.output_dir or self.input_dir) / "cleaning_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("CLEANING RESULTS")
        print("=" * 60)
        print(f"Total images:      {self.stats['total']}")
        print(f"CLIP passed:       {self.stats['clip_passed']} ({100*self.stats['clip_passed']/max(1,self.stats['total']):.1f}%)")
        print(f"CLIP failed:       {self.stats['clip_failed']} ({100*self.stats['clip_failed']/max(1,self.stats['total']):.1f}%)")
        if self.use_fastdup:
            print(f"Fastdup outliers:  {self.stats['fastdup_outliers']}")
        print(f"Errors:            {self.stats['errors']}")
        print("-" * 40)
        print(f"Final dataset:     {self.stats['final']} images")
        print(f"Report saved:      {report_path}")

        # Show sample of removed images
        removed = [p for p, r in self.results.items() if not r["keep"] and not r.get("error")]
        if removed:
            print(f"\nSample removed images (showing first 10):")
            for path in removed[:10]:
                conf = self.results[path]["strawberry_confidence"]
                top = self.results[path]["top_predictions"][0]["label"] if self.results[path]["top_predictions"] else "N/A"
                print(f"  - {path[:50]}... (conf: {conf:.2f}, top: {top})")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Clean strawberry datasets by filtering non-strawberry images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic cleaning
    python scripts/clean_dataset.py --input data/raw --output data/cleaned

    # Stricter threshold
    python scripts/clean_dataset.py --input data/raw --output data/cleaned --threshold 0.8

    # With fastdup outlier detection
    python scripts/clean_dataset.py --input data/raw --output data/cleaned --use-fastdup

    # Dry run to see what would be removed
    python scripts/clean_dataset.py --input data/raw --threshold 0.7 --dry-run
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing images to clean",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for cleaned dataset (required unless --dry-run)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum strawberry confidence to keep image (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be removed, don't copy files",
    )
    parser.add_argument(
        "--use-fastdup",
        action="store_true",
        help="Run fastdup outlier detection after CLIP filtering",
    )
    parser.add_argument(
        "--fastdup-threshold",
        type=float,
        default=0.05,
        help="Fastdup outlier threshold (lower = stricter, default: 0.05)",
    )

    args = parser.parse_args()

    # Validate args
    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry-run is specified")

    if not args.input.exists():
        parser.error(f"Input directory does not exist: {args.input}")

    if args.threshold < 0 or args.threshold > 1:
        parser.error("--threshold must be between 0 and 1")

    # Check dependencies
    check_dependencies()

    # Run cleaner
    cleaner = StrawberryDatasetCleaner(
        input_dir=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        dry_run=args.dry_run,
        use_fastdup=args.use_fastdup,
        fastdup_threshold=args.fastdup_threshold,
    )

    cleaner.run()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
