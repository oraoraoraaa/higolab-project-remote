import requests
import csv
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Modify these paths when moving the script to another location

# Output path: Location where the CSV file will be saved
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Resource', 'Dataset', 'Package-List'))
OUTPUT_FILENAME = "Ruby.csv"

# ============================================================================

def mine_ruby_gems():
    """Mines RubyGems.org to get the whole list of Ruby packages."""
    
    # Fetch gem names from RubyGems API
    print("Fetching gem names from RubyGems API...")
    names_url = "http://rubygems.org/names"
    
    try:
        print("Downloading list of all gem names...")
        response = requests.get(names_url, timeout=120)
        response.raise_for_status()
        gem_names = response.text.strip().split('\n')
        print(f"Found {len(gem_names)} gems")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading gem names: {e}")
        return
    
    # Create the path to the output file
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, OUTPUT_FILENAME)
    
    print("Fetching detailed information for each gem...")
    print("Using parallel processing with 30 workers for faster execution...")
    
    def fetch_gem_info(gem_name):
        """Fetch information for a single gem."""
        gem_info_url = f"https://rubygems.org/api/v1/gems/{gem_name}.json"
        
        homepage_url = "nan"
        repo_url = "nan"
        
        try:
            response = requests.get(gem_info_url, timeout=10)
            if response.status_code == 200:
                gem_info = response.json()
                homepage_url = gem_info.get('homepage_uri', '') or gem_info.get('project_uri', '') or "nan"
                repo_url = gem_info.get('source_code_uri', '') or gem_info.get('homepage_uri', '') or "nan"
                
                # Clean up URLs
                if homepage_url and homepage_url != "nan" and not homepage_url.startswith('http'):
                    homepage_url = "nan"
                if repo_url and repo_url != "nan" and not repo_url.startswith('http'):
                    repo_url = "nan"
                    
        except (requests.exceptions.RequestException, ValueError):
            # If API call fails, continue with nan values
            pass
        
        return gem_name, homepage_url, repo_url
    
    # Use parallel processing to speed up API calls
    results = {}
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        # Submit all tasks
        future_to_gem = {
            executor.submit(fetch_gem_info, gem_name.strip()): (idx, gem_name.strip())
            for idx, gem_name in enumerate(gem_names, start=1)
            if gem_name.strip()
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_gem), total=len(future_to_gem), desc="Processing gems"):
            idx, original_name = future_to_gem[future]
            try:
                gem_name, homepage_url, repo_url = future.result()
                results[idx] = (gem_name, homepage_url, repo_url)
            except Exception as e:
                # If something went wrong, store with nan values
                results[idx] = (original_name, "nan", "nan")
    
    # Write results to CSV in order
    print("Writing results to CSV...")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Platform", "Name", "Homepage URL", "Repository URL"])
        
        for idx in sorted(results.keys()):
            gem_name, homepage_url, repo_url = results[idx]
            writer.writerow([
                idx,
                "RubyGems",
                gem_name,
                homepage_url,
                repo_url,
            ])
    
    print(f"Successfully saved {len(gem_names)} Ruby gems to {output_file}")

if __name__ == "__main__":
    mine_ruby_gems()
