#!/usr/bin/env python3
"""
Script to extract RGEALTI file download links from IGN geoservices pages.
"""

import requests
from bs4 import BeautifulSoup
import re
from typing import List, Set
import time


def fetch_page(url: str, max_retries: int = 3) -> str:
    """
    Fetch HTML content from a URL with retry logic.

    Args:
        url: The URL to fetch
        max_retries: Maximum number of retry attempts

    Returns:
        HTML content as string
    """
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Error fetching {url}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                raise
    return ""


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
    """Main function to process all pages and save links."""

    # Base URL and pages to process
    base_url = "https://geoservices.ign.fr/telechargement-api/RGEALTI"
    pages = [0, 1, 2, 3, 4]

    all_links: Set[str] = set()  # Use set to avoid duplicates

    print("Starting to fetch and parse RGEALTI pages...")

    for page_num in pages:
        url = f"{base_url}?page={page_num}"
        print(f"\nProcessing page {page_num}...")

        try:
            html_content = fetch_page(url)
            links = extract_rgealti_links(html_content)

            print(f"Found {len(links)} links on page {page_num}")
            all_links.update(links)

            # Be nice to the server
            time.sleep(1)

        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
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
