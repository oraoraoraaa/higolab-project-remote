import requests
import csv
import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Modify these paths when moving the script to another location

# Output path: Location where the CSV file will be saved
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Resource', 'Dataset', 'Package-List'))
OUTPUT_FILENAME = "NPM.csv"

# ============================================================================

def fetch_with_retry(url, params=None, timeout=300, max_retries=5, backoff_factor=2):
    """
    Fetch URL with exponential backoff retry logic.
    
    Args:
        url: The URL to fetch
        params: Query parameters
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor for exponential backoff
    
    Returns:
        Response object if successful, None otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout, stream=False)
            response.raise_for_status()
            return response
        except (requests.exceptions.ChunkedEncodingError, 
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                print(f"  Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                raise
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            raise
    return None

def mine_npm_packages():
    """Mines npm registry to get the whole list of npm packages."""
    
    print("Fetching npm package list from registry...")
    print("This will download the complete package database with pagination")
    
    # npm provides an all-docs endpoint that returns all package names
    # Use startkey parameter for pagination (CouchDB-style)
    base_url = "https://replicate.npmjs.com/_all_docs"
    
    package_names = []
    package_names_set = set()  # Use set to track duplicates
    limit = 10000  # Maximum allowed by the API
    startkey = None
    batch_count = 0
    
    try:
        print("Downloading package names from npm registry (paginated)...")
        print("Note: Due to API limitations, we'll fetch the first batch and estimate total count")
        
        # Fetch first batch
        batch_count = 1
        print(f"  Fetching batch {batch_count}...")
        response = fetch_with_retry(base_url, params={'limit': limit}, timeout=300)
        response.raise_for_status()
        data = response.json()
        
        rows = data.get('rows', [])
        total_rows = data.get('total_rows', 0)
        
        print(f"  Total packages in registry: {total_rows:,}")
        
        for row in rows:
            pkg_id = row['id']
            if not pkg_id.startswith('_design/'):
                package_names.append(pkg_id)
                package_names_set.add(pkg_id)
        
        print(f"  Got {len(package_names)} packages from first batch")
        
        # Continue with pagination using startkey
        while len(rows) >= limit:
            batch_count += 1
            last_key = rows[-1]['id']
            
            # Use startkey with the last key we got
            params = {
                'limit': limit,
                'startkey': json.dumps(last_key)
            }
            
            print(f"  Fetching batch {batch_count} starting from '{last_key[:50]}...'")
            response = fetch_with_retry(base_url, params=params, timeout=300)
            response.raise_for_status()
            data = response.json()
            
            rows = data.get('rows', [])
            if not rows:
                break
            
            new_count = 0
            for row in rows:
                pkg_id = row['id']
                # Skip first item if it matches our startkey (it's a duplicate)
                if pkg_id == last_key:
                    continue
                if not pkg_id.startswith('_design/') and pkg_id not in package_names_set:
                    package_names.append(pkg_id)
                    package_names_set.add(pkg_id)
                    new_count += 1
            
            print(f"  Got {new_count} new packages")
            print(f"  Total unique packages collected: {len(package_names):,} / {total_rows:,}")
            
            # If we didn't get any new packages, stop
            if new_count == 0:
                print("  No new packages found, stopping pagination")
                break
        
        print(f"Found {len(package_names)} npm packages in total")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading npm package names: {e}")
        return
    
    # Create the path to the output file
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, OUTPUT_FILENAME)
    
    print("Fetching detailed information for each package...")
    print("Using parallel processing with 40 workers for faster execution...")
    
    def fetch_package_info(package_name):
        """Fetch information for a single npm package."""
        package_info_url = f"https://registry.npmjs.org/{package_name}"
        
        homepage_url = "nan"
        repo_url = "nan"
        
        try:
            response = fetch_with_retry(package_info_url, timeout=10, max_retries=3, backoff_factor=1.5)
            if response and response.status_code == 200:
                package_info = response.json()
                
                # Get homepage URL
                homepage_url = package_info.get('homepage', 'nan')
                if not homepage_url or homepage_url == '':
                    homepage_url = "nan"
                elif not homepage_url.startswith('http'):
                    homepage_url = "nan"
                
                # Get repository URL
                repo_info = package_info.get('repository', {})
                if isinstance(repo_info, dict):
                    repo_url = repo_info.get('url', 'nan')
                elif isinstance(repo_info, str):
                    repo_url = repo_info
                else:
                    repo_url = "nan"
                
                # Keep URLs as-is from the registry (no normalization)
                # Package-Filter will handle normalization
                if not repo_url or repo_url == "nan" or repo_url == "":
                    repo_url = "nan"
                    
        except (requests.exceptions.RequestException, ValueError, KeyError):
            # If API call fails, continue with nan values
            pass
        
        return package_name, homepage_url, repo_url
    
    # Use parallel processing to speed up API calls
    results = {}
    
    with ThreadPoolExecutor(max_workers=40) as executor:
        # Submit all tasks
        future_to_package = {
            executor.submit(fetch_package_info, package_name): (idx, package_name)
            for idx, package_name in enumerate(package_names, start=1)
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
                "NPM",
                package_name,
                homepage_url,
                repo_url,
            ])
    
    print(f"Successfully saved {len(package_names)} npm packages to {output_file}")

if __name__ == "__main__":
    mine_npm_packages()
