import requests
import csv
import os
import gzip
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Modify these paths when moving the script to another location

# Output path: Location where the CSV file will be saved
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Resource', 'Dataset', 'Package-List'))
OUTPUT_FILENAME = "PHP.csv"

# ============================================================================

def download_file(url, filename):
    """Downloads a file from a URL with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte
    
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)

def fetch_package_info(package_name):
    """Fetch information for a single package."""
    package_url = f"https://packagist.org/packages/{package_name}.json"
    
    homepage_url = "nan"
    repo_url = "nan"
    
    try:
        response = requests.get(package_url, timeout=10)
        if response.status_code == 200:
            package_info = response.json()
            
            # Navigate through the JSON structure
            package_data = package_info.get('package', {})
            
            # Get homepage - try multiple sources
            homepage_url = package_data.get('homepage', '') or "nan"
            
            # Get repository URL
            repository = package_data.get('repository', '')
            if repository:
                repo_url = repository
            else:
                # Try to extract from versions
                versions = package_data.get('versions', {})
                if versions:
                    # Get the latest version info
                    for version_key in ['dev-master', 'dev-main', 'master', 'main']:
                        if version_key in versions:
                            version_data = versions[version_key]
                            source = version_data.get('source', {})
                            if source and 'url' in source:
                                repo_url = source['url']
                                break
                    
                    # If still not found, try the first available version
                    if repo_url == "nan" and versions:
                        first_version = next(iter(versions.values()))
                        source = first_version.get('source', {})
                        if source and 'url' in source:
                            repo_url = source['url']
            
            # Clean up URLs
            if homepage_url and homepage_url != "nan" and not homepage_url.startswith('http'):
                homepage_url = "nan"
            if repo_url and repo_url != "nan" and not repo_url.startswith('http'):
                repo_url = "nan"
                
    except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError):
        # If API call fails, continue with nan values
        pass
    
    return package_name, homepage_url, repo_url

def mine_php_packages():
    """Mines Packagist.org to get the whole list of PHP packages."""
    
    # Packagist provides a packages.json file with all package names
    packages_url = "https://packagist.org/packages/list.json"
    
    print("Downloading Packagist package list...")
    
    try:
        response = requests.get(packages_url, timeout=120)
        response.raise_for_status()
        data = response.json()
        package_names = data.get('packageNames', [])
        print(f"Found {len(package_names)} PHP packages")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading package list: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return
    
    if not package_names:
        print("No packages found in the response.")
        return
    
    # Create the path to the output file
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, OUTPUT_FILENAME)
    
    print("Fetching detailed information for each package...")
    print("Using parallel processing with 50 workers for faster execution...")
    
    # Use parallel processing to speed up API calls
    results = {}
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        # Submit all tasks
        future_to_package = {
            executor.submit(fetch_package_info, package_name.strip()): (idx, package_name.strip())
            for idx, package_name in enumerate(package_names, start=1)
            if package_name.strip()
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_package), total=len(future_to_package), desc="Processing packages"):
            idx, original_name = future_to_package[future]
            try:
                package_name, homepage_url, repo_url = future.result()
                results[idx] = (package_name, homepage_url, repo_url)
            except Exception as e:
                # If something went wrong, store with nan values
                results[idx] = (original_name, "nan", "nan")
    
    # Write results to CSV in order
    print("Writing results to CSV...")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Platform", "Name", "Homepage URL", "Repository URL"])
        
        for idx in sorted(results.keys()):
            package_name, homepage_url, repo_url = results[idx]
            writer.writerow([
                idx,
                "Packagist",
                package_name,
                homepage_url,
                repo_url,
            ])
    
    print(f"Successfully saved {len(package_names)} PHP packages to {output_file}")

if __name__ == "__main__":
    mine_php_packages()
