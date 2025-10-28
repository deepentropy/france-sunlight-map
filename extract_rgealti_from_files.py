#!/usr/bin/env python3
"""
Script to extract RGEALTI file download links from saved HTML files.

Usage:
    1. Save the HTML pages from your browser:
       - Visit https://geoservices.ign.fr/telechargement-api/RGEALTI?page=0
       - Right-click -> Save As -> page0.html
       - Repeat for pages 1-4

    2. Run this script:
       python3 extract_rgealti_from_files.py
"""

from bs4 import BeautifulSoup
import re
from typing import List, Set
import os
import glob


def extract_rgealti_links(html_content: str) -> List[str]:
    """
    Extract RGEALTI download links from HTML content.

    Args:
        html_content: HTML content as string

    Returns:
        List of RGEALTI download URLs
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []

    # Find all <a> tags with href attributes
    for link in soup.find_all('a', href=True):
        href = link['href']

        # Look for links containing RGEALTI download patterns
        if 'RGEALTI' in href and ('geopf.fr' in href or 'ign.fr' in href):
            # Match download links (typically .7z, .7z.001, .zip, etc.)
            if re.search(r'\.(7z|zip|tar\.gz|tgz)', href, re.IGNORECASE):
                # Convert relative URLs to absolute if needed
                if href.startswith('http'):
                    links.append(href)
                else:
                    # Handle relative URLs
                    base_url = 'https://data.geopf.fr'
                    if href.startswith('/'):
                        links.append(base_url + href)
                    else:
                        links.append(base_url + '/' + href)

    return links


def main():
    """Main function to process HTML files and save links."""

    # Look for HTML files in current directory
    html_files = glob.glob('*.html')

    if not html_files:
        print("No HTML files found in current directory!")
        print("\nTo use this script:")
        print("1. Visit each page in your browser:")
        for i in range(5):
            print(f"   https://geoservices.ign.fr/telechargement-api/RGEALTI?page={i}")
        print("\n2. Right-click -> Save Page As -> save as 'page0.html', 'page1.html', etc.")
        print("\n3. Run this script again")
        return

    all_links: Set[str] = set()  # Use set to avoid duplicates

    print(f"Found {len(html_files)} HTML file(s) to process...")

    for html_file in sorted(html_files):
        print(f"\nProcessing {html_file}...")

        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()

            links = extract_rgealti_links(html_content)
            print(f"  Found {len(links)} links in {html_file}")
            all_links.update(links)

        except Exception as e:
            print(f"  Error processing {html_file}: {e}")
            continue

    # Save all links to a file
    output_file = "rgealti_links.txt"

    # Sort links for consistent output
    sorted_links = sorted(all_links)

    with open(output_file, 'w') as f:
        for link in sorted_links:
            f.write(link + '\n')

    print(f"\n{'='*60}")
    print(f"Total unique links found: {len(sorted_links)}")
    print(f"Links saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
