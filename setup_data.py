"""
MovieLens Data Setup Script

Downloads and extracts MovieLens 100K dataset.
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_movielens_100k(data_dir: str = '.', force_download: bool = False):
    """
    Download MovieLens 100K dataset.
    
    Parameters
    ----------
    data_dir : str, default '.'
        Directory to download and extract data
    force_download : bool, default False
        If True, re-download even if data already exists
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(data_dir, 'ml-100k.zip')
    extract_dir = os.path.join(data_dir, 'ml-100k')
    
    # Check if already extracted
    if os.path.exists(extract_dir) and os.path.exists(os.path.join(extract_dir, 'u.data')):
        if not force_download:
            print(f"MovieLens 100K data already exists at {extract_dir}")
            print("Set force_download=True to re-download")
            return extract_dir
    
    # Download
    if not os.path.exists(zip_path) or force_download:
        print(f"Downloading MovieLens 100K from {url}...")
        print("This may take a few minutes...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
    else:
        print(f"Zip file already exists at {zip_path}")
    
    # Extract
    if os.path.exists(extract_dir):
        print(f"Removing existing directory {extract_dir}")
        shutil.rmtree(extract_dir)
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # The zip contains ml-100k/ folder, so we might need to move it
    # Check if extraction created nested directory
    nested_path = os.path.join(data_dir, 'ml-100k', 'ml-100k')
    if os.path.exists(nested_path):
        # Move contents up one level
        for item in os.listdir(nested_path):
            src = os.path.join(nested_path, item)
            dst = os.path.join(extract_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        shutil.rmtree(nested_path)
    
    # Validate
    u_data_path = os.path.join(extract_dir, 'u.data')
    u_item_path = os.path.join(extract_dir, 'u.item')
    
    if not os.path.exists(u_data_path):
        raise FileNotFoundError(f"Expected file not found: {u_data_path}")
    if not os.path.exists(u_item_path):
        raise FileNotFoundError(f"Expected file not found: {u_item_path}")
    
    print(f"Successfully extracted MovieLens 100K to {extract_dir}")
    print(f"  - Ratings: {u_data_path}")
    print(f"  - Items: {u_item_path}")
    
    return extract_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download MovieLens 100K dataset')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory to download and extract data (default: current directory)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if data exists'
    )
    
    args = parser.parse_args()
    
    try:
        data_path = download_movielens_100k(args.data_dir, force_download=args.force)
        print(f"\n✓ Setup complete! Data available at: {data_path}")
        print("\nYou can now run the main notebook or scripts.")
    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        raise
