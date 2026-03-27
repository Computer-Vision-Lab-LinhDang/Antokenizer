#!/usr/bin/env python3
"""
HuggingFace Dataset Downloader for MAVT Training

Uses official HuggingFace libraries for efficient downloading:
- huggingface_hub: For direct repository downloads
- datasets: For datasets API with streaming support

Features:
- Progress tracking with tqdm
- Parallel downloads
- Resume support
- Streaming mode for large datasets
- Automatic authentication handling

Usage:
    python download_hf_datasets.py --data-root ./data/datasets
    python download_hf_datasets.py --datasets dfn,webvid,cap3d --parallel 8
    python download_hf_datasets.py --streaming  # Metadata only
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import snapshot_download, login, HfApi
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install huggingface_hub datasets tqdm")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset configurations
DATASETS = {
    "dfn": {
        "repo_id": "apf1/datafilteringnetworks_2b",
        "repo_type": "dataset",
        "description": "DFN-2B: Data Filtering Networks 2B samples",
        "size": "~5TB",
        "dirname": "dfn_2b",
    },
    "webvid": {
        "repo_id": "TempoFunk/webvid-10M",
        "repo_type": "dataset",
        "description": "WebVid-10M: 10M video-text pairs (metadata)",
        "size": "~3GB metadata + videos via URLs",
        "dirname": "webvid_10m",
        "note": "Downloads metadata CSV files only. Use video2dataset for actual videos.",
        "allow_patterns": ["*.csv", "*.json", "*.md", "*.txt"],  # CSV metadata only
    },
    "cap3d": {
        "repo_id": "tiange/Cap3D",
        "repo_type": "dataset",
        "description": "Cap3D: 3D captions for Objaverse",
        "size": "~200GB (captions + point clouds)",
        "dirname": "cap3d",
    },
    "textocr": {
        "repo_id": "facebook/textocr",
        "repo_type": "dataset",
        "description": "TextOCR: Text recognition dataset",
        "size": "~50GB",
        "dirname": "textocr",
        "use_datasets_api": True,
    },
    "objaverse": {
        "repo_id": "allenai/objaverse",
        "repo_type": "dataset",
        "description": "Objaverse: 3D object metadata",
        "size": "~5GB metadata",
        "dirname": "objaverse_xl",
        "note": "Metadata only. Use objaverse library for 3D objects.",
    },
    "imagenet": {
        "repo_id": "imagenet-1k",
        "repo_type": "dataset",
        "description": "ImageNet-1K (gated - requires auth)",
        "size": "~150GB",
        "dirname": "imagenet_1k",
        "use_datasets_api": True,
        "gated": True,
    },
}


class HFDatasetDownloader:
    """Download HuggingFace datasets with progress tracking."""

    def __init__(
        self,
        data_root: str,
        cache_dir: Optional[str] = None,
        parallel: int = 4,
        streaming: bool = False,
        token: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        os.environ["HF_HOME"] = self.cache_dir

        self.parallel = parallel
        self.streaming = streaming
        self.token = token

        # Authenticate if token provided
        if self.token:
            logger.info("Authenticating with HuggingFace...")
            login(token=self.token, add_to_git_credential=True)

    def download_dataset(self, dataset_name: str) -> bool:
        """Download a single dataset."""
        if dataset_name not in DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            logger.info(f"Available datasets: {', '.join(DATASETS.keys())}")
            return False

        config = DATASETS[dataset_name]
        logger.info("=" * 70)
        logger.info(f"Downloading: {config['description']}")
        logger.info(f"Size: {config['size']}")
        if "note" in config:
            logger.info(f"Note: {config['note']}")
        logger.info("=" * 70)

        local_dir = self.data_root / config["dirname"]
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            if config.get("use_datasets_api"):
                return self._download_via_datasets_api(config, local_dir)
            else:
                return self._download_via_hub(config, local_dir)
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False

    def _download_via_hub(self, config: dict, local_dir: Path) -> bool:
        """Download using huggingface_hub (direct repo download)."""
        repo_id = config["repo_id"]
        repo_type = config.get("repo_type", "dataset")

        logger.info(f"Downloading {repo_id} to {local_dir}...")

        # Check if gated
        if config.get("gated") and not self.token:
            logger.warning(
                f"{repo_id} is gated and requires authentication. "
                "Use --token or 'huggingface-cli login'"
            )

        try:
            # Get allow_patterns from config or use defaults
            allow_patterns = config.get("allow_patterns")

            # Legacy fallback for specific datasets
            if not allow_patterns:
                if config.get("dirname") == "webvid_10m":
                    allow_patterns = ["*.csv", "*.json", "*.parquet", "*.txt", "*.md"]
                elif config.get("dirname") == "cap3d":
                    allow_patterns = ["*.csv", "*.md", "*.txt", "*.json"]

            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=str(local_dir),
                allow_patterns=allow_patterns,
                max_workers=self.parallel,
                resume_download=True,
                local_dir_use_symlinks=False,
            )

            logger.info(f"✓ Downloaded to {local_dir}")
            return True

        except Exception as e:
            logger.error(f"Error: {e}")
            return False

    def _download_via_datasets_api(self, config: dict, local_dir: Path) -> bool:
        """Download using datasets library (streaming or full)."""
        repo_id = config["repo_id"]

        logger.info(f"Loading {repo_id}...")

        try:
            if self.streaming:
                # Streaming mode - don't download, just create streaming dataset
                logger.info("Using streaming mode (no download)")
                try:
                    dataset = load_dataset(repo_id, streaming=True)
                    logger.info(f"✓ {repo_id} ready for streaming")

                    # Save a sample for verification
                    sample_file = local_dir / "sample.json"
                    try:
                        import json
                        split = next(iter(dataset.keys()))
                        sample = next(iter(dataset[split].take(1)))
                        with open(sample_file, 'w') as f:
                            json.dump(str(sample), f, indent=2)
                        logger.info(f"Sample saved to {sample_file}")
                    except Exception as e:
                        logger.warning(f"Could not save sample: {e}")
                except Exception as e:
                    logger.warning(f"Streaming not available, falling back to hub download: {e}")
                    return self._download_via_hub(config, local_dir)

            else:
                # Full download
                logger.info("Downloading full dataset...")
                try:
                    dataset = load_dataset(
                        repo_id,
                        cache_dir=self.cache_dir,
                        num_proc=self.parallel,
                    )

                    # Save to disk in arrow format
                    logger.info(f"Saving to {local_dir}...")
                    dataset.save_to_disk(str(local_dir))
                    logger.info(f"✓ Saved to {local_dir}")
                except Exception as e:
                    logger.warning(f"Datasets API failed, trying hub download: {e}")
                    return self._download_via_hub(config, local_dir)

            return True

        except Exception as e:
            logger.error(f"Error: {e}")
            return False

    def list_available_datasets(self):
        """List all available datasets."""
        print("\n" + "=" * 70)
        print("Available HuggingFace Datasets for MAVT")
        print("=" * 70 + "\n")

        for name, config in DATASETS.items():
            print(f"📦 {name}")
            print(f"   Repo: {config['repo_id']}")
            print(f"   Description: {config['description']}")
            print(f"   Size: {config['size']}")
            if "note" in config:
                print(f"   Note: {config['note']}")
            if config.get("gated"):
                print(f"   ⚠️  Gated - requires authentication")
            print()

    def download_multiple(self, dataset_names: List[str]) -> dict:
        """Download multiple datasets and return status."""
        results = {}

        for name in dataset_names:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing: {name} ({dataset_names.index(name)+1}/{len(dataset_names)})")
            logger.info(f"{'='*70}\n")

            success = self.download_dataset(name)
            results[name] = success

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets for MAVT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download single dataset
  python download_hf_datasets.py --datasets webvid

  # Download multiple datasets
  python download_hf_datasets.py --datasets dfn,webvid,cap3d

  # Use streaming mode (metadata only)
  python download_hf_datasets.py --datasets webvid --streaming

  # With authentication for gated datasets
  python download_hf_datasets.py --datasets imagenet --token YOUR_TOKEN

  # List available datasets
  python download_hf_datasets.py --list
        """
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/datasets",
        help="Root directory for datasets (default: ./data/datasets)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default="webvid,cap3d",
        help="Comma-separated list of datasets to download"
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)"
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (no full download, saves space)"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for gated datasets (or set HF_TOKEN env var)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and exit"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = HFDatasetDownloader(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        parallel=args.parallel,
        streaming=args.streaming,
        token=args.token,
    )

    # List datasets if requested
    if args.list:
        downloader.list_available_datasets()
        return

    # Parse dataset list
    dataset_names = [d.strip() for d in args.datasets.split(",")]

    logger.info("=" * 70)
    logger.info("HuggingFace Dataset Downloader for MAVT")
    logger.info("=" * 70)
    logger.info(f"Data root: {Path(args.data_root).absolute()}")
    logger.info(f"Cache dir: {args.cache_dir or '~/.cache/huggingface'}")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info(f"Parallel: {args.parallel}")
    logger.info(f"Streaming: {args.streaming}")
    logger.info("=" * 70 + "\n")

    # Download datasets
    results = downloader.download_multiple(dataset_names)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Download Summary")
    logger.info("=" * 70 + "\n")

    success_count = sum(results.values())
    total_count = len(results)

    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {name}")

    logger.info(f"\nCompleted: {success_count}/{total_count} datasets")
    logger.info(f"Data root: {Path(args.data_root).absolute()}")

    if success_count < total_count:
        logger.warning("\nSome downloads failed. Check logs above for details.")
        sys.exit(1)

    logger.info("\n🎉 All downloads complete!")
    logger.info("\nNext steps:")
    logger.info("  1. For WebVid videos: Use video2dataset with metadata")
    logger.info("  2. For Objaverse 3D: Use objaverse Python library")
    logger.info("  3. Start training:")
    logger.info(f"     python -m train.example_stage_training --data-root {args.data_root}")


if __name__ == "__main__":
    main()
