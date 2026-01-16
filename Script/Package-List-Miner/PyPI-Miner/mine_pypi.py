import requests
import csv
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Modify these paths when moving the script to another location

# Output path: Location where the CSV file will be saved
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Resource', 'Dataset', 'Package-List'))
OUTPUT_FILENAME = "PyPI.csv"

# ============================================================================

def mine_pypi_packages():
    """Mines PyPI to get the whole list of Python packages."""
    
    print("Fetching PyPI package list...")
    print("Using PyPI's simple API to get all package names")
    
    # PyPI simple API provides a list of all packages
    simple_index_url = "https://pypi.org/simple/"
    
    try:
        print("Downloading package names from PyPI...")
        response = requests.get(simple_index_url, timeout=120)
        response.raise_for_status()
        
        # Parse HTML to extract package names
        # The simple index returns HTML with links to each package
        import re
        package_pattern = re.compile(r'<a[^>]*>([^<]+)</a>')
        package_names = package_pattern.findall(response.text)
        
        print(f"Found {len(package_names)} PyPI packages")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PyPI package names: {e}")
        return
    
    # Create the path to the output file
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, OUTPUT_FILENAME)
    
    print("Fetching detailed information for each package...")
    print("Using parallel processing with 50 workers for faster execution...")
    
    def fetch_package_info(package_name):
        """Fetch information for a single PyPI package."""
        # Use PyPI JSON API for package metadata
        package_info_url = f"https://pypi.org/pypi/{package_name}/json"
        
        homepage_url = "nan"
        repo_url = "nan"
        
        try:
            response = requests.get(package_info_url, timeout=10)
            if response.status_code == 200:
                package_info = response.json()
                info = package_info.get('info', {})
                
                # Get homepage URL
                homepage_url = info.get('home_page', '') or info.get('package_url', '') or "nan"
                if not homepage_url or homepage_url == '':
                    homepage_url = "nan"
                elif not homepage_url.startswith('http'):
                    homepage_url = "nan"
                
                # Get repository URL - try multiple fields
                project_urls = info.get('project_urls', {})
                if project_urls:
                    # Try common repository field names
                    for key in ['Source', 'Source Code', 'Repository', 'Code', 'GitHub', 'GitLab']:
                        if key in project_urls:
                            repo_url = project_urls[key]
                            break
                
                # If not found in project_urls, try home_page if it looks like a repo
                if repo_url == "nan":
                    home_page = info.get('home_page', '')
                    if home_page and ('github.com' in home_page or 'gitlab.com' in home_page or 'bitbucket.org' in home_page):
                        repo_url = home_page
                
                # Clean up repository URL
                if repo_url and repo_url != "nan":
                    if not repo_url.startswith('http'):
                        repo_url = "nan"
                else:
                    repo_url = "nan"
                    
        except (requests.exceptions.RequestException, ValueError, KeyError):
            # If API call fails, continue with nan values
            pass
        
        return package_name, homepage_url, repo_url
    
    # Use parallel processing to speed up API calls
    results = {}
    
    with ThreadPoolExecutor(max_workers=50) as executor:
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
                "PyPI",
                package_name,
                homepage_url,
                repo_url,
            ])
    
    print(f"Successfully saved {len(package_names)} PyPI packages to {output_file}")

if __name__ == "__main__":
    mine_pypi_packages()
