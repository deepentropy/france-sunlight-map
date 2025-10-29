#!/usr/bin/env python3
"""
Download RGEALTI files filtered by department and resolution.

Usage:
    python3 download_rgealti.py --department D001 --resolution 5M
    python3 download_rgealti.py --department D972 --resolution 1M
    python3 download_rgealti.py --list-departments
    python3 download_rgealti.py --list-resolutions
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Set


def read_links_file(links_file: str = "links/rgealti_links.txt") -> List[str]:
    """Read all links from the links file."""
    try:
        with open(links_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Links file '{links_file}' not found!")
        print("Please run the extraction script first to generate the links file.")
        sys.exit(1)


def parse_link_info(link: str) -> Dict[str, str]:
    """
    Parse information from a RGEALTI link.

    Example link:
    https://data.geopf.fr/telechargement/download/RGEALTI/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D001_2023-08-08/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D001_2023-08-08.7z

    Returns:
        Dictionary with 'department', 'resolution', 'full_name', 'url'
    """
    # Extract the filename pattern: RGEALTI_X-X_XM_..._DXXX_...
    pattern = r'RGEALTI_[\d-]+_(1M|5M)_.*?_(D\d+)_'
    match = re.search(pattern, link)

    if match:
        resolution = match.group(1)
        department = match.group(2)

        # Extract filename from URL
        filename = link.split('/')[-1]

        return {
            'department': department,
            'resolution': resolution,
            'filename': filename,
            'url': link
        }

    return None


def filter_links(links: List[str], department: str = None, resolution: str = None) -> List[Dict[str, str]]:
    """
    Filter links by department and/or resolution.

    Args:
        links: List of URLs
        department: Department code (e.g., 'D001', 'D972')
        resolution: Resolution (e.g., '1M', '5M')

    Returns:
        List of parsed link info dictionaries
    """
    filtered = []

    for link in links:
        info = parse_link_info(link)
        if not info:
            continue

        # Apply filters
        if department and info['department'] != department.upper():
            continue

        if resolution and info['resolution'] != resolution.upper():
            continue

        filtered.append(info)

    return filtered


def get_unique_departments(links: List[str]) -> Set[str]:
    """Extract all unique department codes from links."""
    departments = set()
    for link in links:
        info = parse_link_info(link)
        if info:
            departments.add(info['department'])
    return sorted(departments)


def get_unique_resolutions(links: List[str]) -> Set[str]:
    """Extract all unique resolutions from links."""
    resolutions = set()
    for link in links:
        info = parse_link_info(link)
        if info:
            resolutions.add(info['resolution'])
    return sorted(resolutions)


def download_files(filtered_links: List[Dict[str, str]], output_dir: str, method: str = "wget"):
    """
    Download filtered files.

    Args:
        filtered_links: List of link info dictionaries
        output_dir: Output directory for downloads
        method: Download method ('wget', 'curl', or 'aria2c')
    """
    if not filtered_links:
        print("No files match the specified filters.")
        return

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(filtered_links)} file(s) to download")
    print(f"Output directory: {output_dir}")
    print(f"Download method: {method}\n")

    # Show files to be downloaded
    print("Files to download:")
    for i, link_info in enumerate(filtered_links, 1):
        print(f"{i}. {link_info['filename']}")

    # Confirm download
    response = input(f"\nProceed with download? [y/N]: ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    # Create a temporary file with URLs
    urls_file = os.path.join(output_dir, ".download_urls.txt")
    with open(urls_file, 'w') as f:
        for link_info in filtered_links:
            f.write(link_info['url'] + '\n')

    print(f"\nStarting download...\n")

    # Download based on method
    try:
        if method == "wget":
            cmd = ["wget", "-i", urls_file, "-P", output_dir, "--continue", "--progress=bar:force"]
            subprocess.run(cmd, check=True)

        elif method == "curl":
            # Download files one by one with curl
            for link_info in filtered_links:
                output_file = os.path.join(output_dir, link_info['filename'])
                cmd = ["curl", "-L", "-o", output_file, "-C", "-", link_info['url']]
                subprocess.run(cmd, check=True)

        elif method == "aria2c":
            cmd = ["aria2c", "-i", urls_file, "-d", output_dir, "-c", "-j", "3", "-x", "3"]
            subprocess.run(cmd, check=True)

        else:
            print(f"Unknown download method: {method}")
            return

        print(f"\n{'='*60}")
        print(f"Download complete!")
        print(f"Files saved to: {output_dir}")
        print(f"{'='*60}")

    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")
    except FileNotFoundError:
        print(f"Error: '{method}' command not found. Please install it first.")
        print(f"  - wget: sudo apt-get install wget")
        print(f"  - curl: sudo apt-get install curl")
        print(f"  - aria2c: sudo apt-get install aria2")
    finally:
        # Clean up temporary file
        if os.path.exists(urls_file):
            os.remove(urls_file)


def main():
    parser = argparse.ArgumentParser(
        description="Download RGEALTI files filtered by department and resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 5M resolution files for department D001
  python3 download_rgealti.py --department D001 --resolution 5M

  # Download all 1M resolution files for department D972
  python3 download_rgealti.py --department D972 --resolution 1M

  # Download all files for department D001 (any resolution)
  python3 download_rgealti.py --department D001

  # List all available departments
  python3 download_rgealti.py --list-departments

  # List all available resolutions
  python3 download_rgealti.py --list-resolutions
        """
    )

    parser.add_argument(
        "--department",
        type=str,
        help="Department code (e.g., D001, D972)"
    )

    parser.add_argument(
        "--resolution",
        type=str,
        choices=['1M', '5M', '1m', '5m'],
        help="Resolution: 1M or 5M"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="RGEALTI",
        help="Output directory for downloads (default: RGEALTI)"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=['wget', 'curl', 'aria2c'],
        default='wget',
        help="Download method (default: wget)"
    )

    parser.add_argument(
        "--links-file",
        type=str,
        default="links/rgealti_links.txt",
        help="Path to links file (default: links/rgealti_links.txt)"
    )

    parser.add_argument(
        "--list-departments",
        action="store_true",
        help="List all available departments"
    )

    parser.add_argument(
        "--list-resolutions",
        action="store_true",
        help="List all available resolutions"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )

    args = parser.parse_args()

    # Read links file
    links = read_links_file(args.links_file)

    if not links:
        print("No links found in the links file!")
        sys.exit(1)

    # List departments
    if args.list_departments:
        departments = get_unique_departments(links)
        print(f"Available departments ({len(departments)}):")
        for dept in departments:
            print(f"  - {dept}")
        sys.exit(0)

    # List resolutions
    if args.list_resolutions:
        resolutions = get_unique_resolutions(links)
        print(f"Available resolutions ({len(resolutions)}):")
        for res in resolutions:
            print(f"  - {res}")
        sys.exit(0)

    # Filter links
    resolution = args.resolution.upper() if args.resolution else None
    filtered_links = filter_links(links, args.department, resolution)

    if not filtered_links:
        print("No files match the specified filters.")
        print("\nUse --list-departments and --list-resolutions to see available options.")
        sys.exit(1)

    # Organize output directory by department and resolution
    if args.department and args.resolution:
        output_dir = os.path.join(args.output_dir, f"{args.department}_{args.resolution}")
    elif args.department:
        output_dir = os.path.join(args.output_dir, args.department)
    elif args.resolution:
        output_dir = os.path.join(args.output_dir, args.resolution)
    else:
        output_dir = args.output_dir

    # Dry run
    if args.dry_run:
        print(f"Would download {len(filtered_links)} file(s) to {output_dir}:\n")
        for i, link_info in enumerate(filtered_links, 1):
            print(f"{i}. {link_info['filename']}")
            print(f"   Department: {link_info['department']}, Resolution: {link_info['resolution']}")
        sys.exit(0)

    # Download files
    download_files(filtered_links, output_dir, args.method)


if __name__ == "__main__":
    main()
