#!/usr/bin/env python3
"""
Download Habitat datasets for Vesper.

Usage:
    python scripts/download_datasets.py --dataset hssd-hab
    python scripts/download_datasets.py --list
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vesper.utils.dataset import (
    download_dataset,
    list_available_datasets,
    verify_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Download Habitat datasets for Vesper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --list                    List available datasets
    %(prog)s --dataset hssd-hab        Download HSSD dataset
    %(prog)s --dataset replica-cad     Download ReplicaCAD dataset
    %(prog)s --verify hssd-hab         Verify dataset installation
        """,
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to download",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for datasets",
    )
    
    parser.add_argument(
        "--verify",
        type=str,
        help="Verify a dataset is installed",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        print("\nAvailable Datasets:\n")
        print("-" * 60)
        for name, info in list_available_datasets().items():
            print(f"\n  {name}")
            print(f"    Description: {info['description']}")
            print(f"    Size: {info['size']}")
        print("\n" + "-" * 60)
        return 0
    
    # Verify dataset
    if args.verify:
        print(f"\nVerifying dataset: {args.verify}")
        if verify_dataset(args.verify):
            print("✓ Dataset is installed")
            return 0
        else:
            print("✗ Dataset not found")
            return 1
    
    # Download dataset
    if args.dataset:
        print(f"\n{'=' * 60}")
        print(f"Downloading: {args.dataset}")
        print(f"{'=' * 60}\n")
        
        success = download_dataset(
            args.dataset,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
        
        if success:
            print(f"\n✓ Dataset '{args.dataset}' downloaded successfully!")
            return 0
        else:
            print(f"\n✗ Failed to download '{args.dataset}'")
            return 1
    
    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
