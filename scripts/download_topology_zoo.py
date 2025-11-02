"""Download and extract networks from Internet Topology Zoo."""

import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup


def scrape_graphml_links(
    dataset_url: str = "https://topology-zoo.org/dataset.html",
) -> list[str]:
    """Scrape GraphML file links from the Topology Zoo dataset page.

    Args:
        dataset_url: URL of the dataset page

    Returns:
        List of GraphML file URLs
    """
    print(f"Scraping GraphML links from {dataset_url}...")
    try:
        response = requests.get(dataset_url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to {dataset_url}: {e}")
        print("Please check your internet connection and try again.")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    # Find all links that end with .graphml
    graphml_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".graphml"):
            # Convert relative URLs to absolute
            # Links are typically: files/Agis.graphml or just Agis.graphml
            if href.startswith("http"):
                full_url = href
            elif href.startswith("/"):
                full_url = f"https://topology-zoo.org{href}"
            elif href.startswith("files/"):
                full_url = f"https://topology-zoo.org/{href}"
            else:
                # Assume it's just the filename, prepend files/
                full_url = f"https://topology-zoo.org/files/{href}"
            graphml_links.append(full_url)

    # Remove duplicates while preserving order
    seen = set()
    unique_links = []
    for link in graphml_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    print(f"Found {len(unique_links)} GraphML file links")
    return unique_links


def download_topology_zoo(output_dir: Path, overwrite: bool = False):
    """Download Internet Topology Zoo dataset.

    Args:
        output_dir: Directory to save downloaded files
        overwrite: Whether to overwrite existing files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create graphml directory
    graphml_dir = output_dir / "graphml"
    graphml_dir.mkdir(exist_ok=True)

    # Scrape GraphML links from the website
    graphml_urls = scrape_graphml_links()

    if not graphml_urls:
        print("No GraphML links found. Please check the website manually.")
        return

    # Download each GraphML file
    print(f"\nDownloading {len(graphml_urls)} GraphML files...")
    downloaded = 0
    skipped = 0
    failed = 0

    for url in tqdm(graphml_urls, desc="Downloading", unit="file"):
        # Extract filename from URL
        filename = url.split("/")[-1]
        filepath = graphml_dir / filename

        # Skip if file exists and not overwriting
        if filepath.exists() and not overwrite:
            skipped += 1
            continue

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Verify it's actually a GraphML file (check for XML/GraphML content)
            content = response.content[:100].decode("utf-8", errors="ignore")
            if "graphml" in content.lower() or "xml" in content.lower():
                filepath.write_bytes(response.content)
                downloaded += 1
            else:
                print(
                    f"\nWarning: {filename} doesn't appear to be a valid GraphML file, skipping"
                )
                failed += 1
        except Exception as e:
            print(f"\nError downloading {filename}: {e}")
            failed += 1

    print(f"\nDownload summary:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")

    # Count final GraphML files
    final_graphml_files = list(graphml_dir.glob("*.graphml"))
    print(f"\nTopology Zoo dataset ready in {output_dir}")
    print(f"GraphML files location: {graphml_dir}")
    print(f"Total GraphML files available: {len(final_graphml_files)}")

    if len(final_graphml_files) == 0:
        print(f"\nNo GraphML files found. Please check the website manually:")
        print(f"Visit: https://topology-zoo.org/dataset.html")


def main():
    parser = argparse.ArgumentParser(
        description="Download Internet Topology Zoo networks"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/topology_zoo",
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    download_topology_zoo(output_dir, args.overwrite)


if __name__ == "__main__":
    main()
