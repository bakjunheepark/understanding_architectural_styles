"""
Download hybrid building images for architectural style evaluation.

This script was used to collect images of buildings that intentionally combine
multiple architectural styles. Images were collected using DuckDuckGo image search.

Collection Process:
1. Buildings were curated from scholarly sources documenting architectural hybrids
2. Each building has 2 images showing clear exterior views
3. Images with watermarks were manually removed and replaced with clean alternatives
4. Images are center-cropped to 299x299 pixels (model input size)

Usage:
    python download_hybrid_images.py

The pre-collected dataset is included in hybrid_buildings/ - this script is
provided for reproducibility and documentation purposes.
"""

import csv
import os
import time
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image

try:
    from duckduckgo_search import DDGS
except ImportError:
    print("Please install duckduckgo_search: pip install duckduckgo_search")
    exit(1)


# Configuration
CSV_PATH = "hybrid_buildings/buildings.csv"
OUTPUT_DIR = "hybrid_buildings"
IMAGE_SIZE = 299  # Target size for model
IMAGES_PER_BUILDING = 2
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_SEARCHES = 2  # seconds, to avoid rate limiting


def create_output_dir():
    """Create output directory if it doesn't exist."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}/")


def load_buildings():
    """Load buildings from CSV file."""
    buildings = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            buildings.append({
                'name': row['building'],
                'file_1': row['file_1'],
                'file_2': row['file_2'],
                'style_1': row['style_1'],
                'style_2': row['style_2'],
            })
    return buildings


def get_custom_search_term(building_name: str) -> str:
    """Get custom search term for buildings with special characters."""
    custom_terms = {
        'Sagrada Família': 'Sagrada Familia Barcelona',
        'Sacré-Cœur Basilica': 'Sacre Coeur Paris',
        'Palau de la Música Catalana': 'Palau de la Musica Catalana Barcelona',
        '550 Madison Ave': '550 Madison Avenue New York AT&T Building',
    }
    return custom_terms.get(building_name, building_name)


def search_images(query: str, num_images: int = 2) -> list:
    """Search for images using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(
                query,
                region="wt-wt",
                safesearch="moderate",
                size="Large",
                max_results=num_images + 5
            ))
        return [r['image'] for r in results]
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def download_image(url: str) -> Image.Image | None:
    """Download image from URL and return PIL Image."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"  Download error: {e}")
        return None


def crop_to_square(img: Image.Image, target_size: int = 299) -> Image.Image:
    """Center crop image to square and resize to target size."""
    width, height = img.size
    
    if width > height:
        left = (width - height) // 2
        top = 0
        right = left + height
        bottom = height
    else:
        left = 0
        top = (height - width) // 2
        right = width
        bottom = top + width
    
    img = img.crop((left, top, right, bottom))
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return img


def save_image(img: Image.Image, filename: str):
    """Save image to output directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath, 'JPEG', quality=95)
    return filepath


def process_building(building: dict) -> tuple[bool, bool]:
    """Download and process images for a single building."""
    name = building['name']
    file_1 = building['file_1']
    file_2 = building['file_2']
    
    # Check if both files already exist
    file_1_path = os.path.join(OUTPUT_DIR, file_1)
    file_2_path = os.path.join(OUTPUT_DIR, file_2)
    file_1_exists = os.path.exists(file_1_path)
    file_2_exists = os.path.exists(file_2_path)
    
    if file_1_exists and file_2_exists:
        print(f"\n[BUILDING] {name} - SKIPPING (both images exist)")
        return True, True
    
    print(f"\n[BUILDING] {name}")
    if file_1_exists:
        print(f"  Already have: {file_1}")
    if file_2_exists:
        print(f"  Already have: {file_2}")
    
    # Build search query
    search_term = get_custom_search_term(name)
    search_query = f"{search_term} architecture exterior building photograph"
    print(f"  Searching: {search_query}")
    
    # Search for images
    image_urls = search_images(search_query, IMAGES_PER_BUILDING)
    
    if not image_urls:
        print(f"  [X] No images found")
        return file_1_exists, file_2_exists
    
    print(f"  Found {len(image_urls)} image URLs")
    
    success_1 = file_1_exists
    success_2 = file_2_exists
    
    for url in image_urls:
        need_file_1 = not success_1
        need_file_2 = not success_2
        
        if not need_file_1 and not need_file_2:
            break
            
        img = download_image(url)
        if img is None:
            continue
        
        if img.width < 200 or img.height < 200:
            print(f"  Skipping small image: {img.width}x{img.height}")
            continue
        
        img = crop_to_square(img, IMAGE_SIZE)
        
        if need_file_1:
            save_image(img, file_1)
            print(f"  [OK] Saved: {file_1} ({IMAGE_SIZE}x{IMAGE_SIZE})")
            success_1 = True
        elif need_file_2:
            save_image(img, file_2)
            print(f"  [OK] Saved: {file_2} ({IMAGE_SIZE}x{IMAGE_SIZE})")
            success_2 = True
    
    return success_1, success_2


def main():
    print("=" * 60)
    print("Hybrid Building Image Downloader")
    print("=" * 60)
    
    create_output_dir()
    buildings = load_buildings()
    print(f"Loaded {len(buildings)} buildings from {CSV_PATH}\n")
    
    success_count = 0
    partial_count = 0
    failed_count = 0
    
    for i, building in enumerate(buildings, 1):
        print(f"[{i}/{len(buildings)}]", end="")
        
        success_1, success_2 = process_building(building)
        
        if success_1 and success_2:
            success_count += 1
        elif success_1 or success_2:
            partial_count += 1
        else:
            failed_count += 1
        
        # Rate limiting delay
        time.sleep(DELAY_BETWEEN_SEARCHES)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total buildings: {len(buildings)}")
    print(f"[OK] Complete (2 images): {success_count}")
    print(f"[!] Partial (1 image):   {partial_count}")
    print(f"[X] Failed (0 images):   {failed_count}")
    print(f"\nImages saved to: {OUTPUT_DIR}/")
    print("\nNote: Manually review images and remove any with watermarks.")
    print("Re-run script to download replacements for removed images.")


if __name__ == "__main__":
    main()

