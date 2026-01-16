import requests
import csv
import os
import sys
from tqdm import tqdm
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gzip
from io import BytesIO
import argparse


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Modify these paths when moving the script to another location

# Default output path: Location where the CSV file will be saved
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Resource', 'Dataset', 'Package-List'))
DEFAULT_OUTPUT_FILENAME = "Go.csv"

# ============================================================================

def create_session():
    """Creates a requests session with connection pooling and retries."""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,  # Increased for better performance
        pool_maxsize=40
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Enable compression
    session.headers.update({
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'Go-Module-Miner/1.0'
    })
    
    return session

def download_go_index():
    """Downloads the Go module index from the official proxy with optimized fetching."""
    
    base_url = "https://index.golang.org/index"
    
    print("Downloading Go module index...")
    print("This may take a while as the index is continuously updated...")
    print("Using optimized fetching with proper pagination...")
    print("Note: The index contains ALL versions of modules, so we deduplicate by module path")
    print("Estimated time: 20-40 minutes for complete download")
    print("You can monitor progress in the progress bar below.")
    
    modules_set = set()  # Use set for faster deduplication
    since = ""
    batch_count = 0
    session = create_session()
    consecutive_empty_batches = 0
    total_entries = 0
    
    # Checkpoint every N batches to save progress
    checkpoint_interval = 1000
    checkpoint_file = os.path.join(os.path.dirname(__file__), '.checkpoint.json')
    
    # Try to load from checkpoint
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                modules_set = set(checkpoint['modules'])
                since = checkpoint['since']
                batch_count = checkpoint['batch_count']
                total_entries = checkpoint['total_entries']
                print(f"Resuming from checkpoint: batch {batch_count}, {len(modules_set)} modules")
        except:
            print("Could not load checkpoint, starting fresh")
    
    try:
        # Use a simple ascii progress bar written to stdout with a fixed width
        # This reduces issues where unicode/auto-sizing causes the bar to print
        # a new line on each update in some terminals.
        pbar = tqdm(
            desc="Fetching batches",
            unit="batch",
            initial=batch_count,
            file=sys.stdout,
            ascii=True,
            ncols=100,
            leave=True,
            dynamic_ncols=False,
        )
        
        while True:
            batch_count += 1
            
            url = f"{base_url}?since={since}" if since else base_url
            response = session.get(url, timeout=60)
            response.raise_for_status()
            
            text = response.text.strip()
            if not text:
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= 3:
                    break
                continue
            
            consecutive_empty_batches = 0
            lines = text.split('\n')
            if not lines or (len(lines) == 1 and not lines[0].strip()):
                break
            
            # Batch process JSON lines - parse only what we need
            new_modules = 0
            last_timestamp = since
            entries_processed = 0
            
            for line in lines:
                if not line.strip():
                    continue
                try:
                    # Find the path and timestamp without full JSON parse for speed
                    # Only parse if we might need it
                    entry = json.loads(line)
                    entries_processed += 1
                    module_path = entry.get('Path')
                    if module_path and module_path not in modules_set:
                        modules_set.add(module_path)
                        new_modules += 1
                    
                    # Always update timestamp to the last entry's timestamp
                    ts = entry.get('Timestamp')
                    if ts:
                        last_timestamp = ts
                except (json.JSONDecodeError, KeyError):
                    continue
            
            total_entries += entries_processed
            
            # Only update 'since' if we got a new timestamp
            if last_timestamp != since:
                since = last_timestamp
            else:
                # If timestamp didn't change, we might be stuck
                pbar.write(f"\nWarning: Timestamp didn't change in batch {batch_count}, stopping")
                break
            
            # Save checkpoint periodically
            if batch_count % checkpoint_interval == 0:
                try:
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'modules': list(modules_set),
                            'since': since,
                            'batch_count': batch_count,
                            'total_entries': total_entries
                        }, f)
                except:
                    pass  # Don't fail if checkpoint save fails
            
            # Update progress bar - set to current batch count and update postfix
            pbar.update(1)
            pbar.set_postfix(unique=f"{len(modules_set):,}", new=new_modules, total=f"{total_entries:,}")
            
            # If we got less than 2000 entries, we're likely at the end
            # Note: The API returns up to 2000 entries per request
            if entries_processed < 2000:
                pbar.write(f"\nReceived {entries_processed} entries (less than 2000), reached end of index")
                break
        
        pbar.close()
        
        print(f"\nFetched {batch_count} batches, {total_entries:,} total entries")
        
        # Clean up checkpoint file on successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
    except KeyboardInterrupt:
        if 'pbar' in locals():
            pbar.close()
        print(f"\n\nInterrupted! Progress saved to checkpoint.")
        print(f"Run the script again to resume from batch {batch_count}")
        # Save final checkpoint
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'modules': list(modules_set),
                    'since': since,
                    'batch_count': batch_count,
                    'total_entries': total_entries
                }, f)
        except:
            pass
        raise
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading Go module index: {e}")
        return []
    finally:
        session.close()
    
    # Convert set to list of dicts
    modules = [{'path': path} for path in modules_set]
    return modules

def get_module_info(module_path, session):
    """Fetches module information from the Go proxy API."""
    
    homepage_url = "nan"
    repo_url = "nan"
    
    try:
        # Use the Go proxy API to get the latest version info
        # First, get the list of versions
        versions_url = f"https://proxy.golang.org/{module_path}/@v/list"
        response = session.get(versions_url, timeout=10)
        
        latest_version = None
        
        if response.status_code == 200:
            versions = response.text.strip().split('\n')
            if versions and versions[0]:
                # Use the latest version
                latest_version = versions[-1]
        
        # Try @v/latest first to get both version and repository URL
        latest_url = f"https://proxy.golang.org/{module_path}/@latest"
        latest_response = session.get(latest_url, timeout=10)
        if latest_response.status_code == 200:
            latest_data = latest_response.json()
            if not latest_version:
                latest_version = latest_data.get('Version')
            # Extract repository URL from Origin field if available
            origin = latest_data.get('Origin', {})
            if origin:
                origin_url = origin.get('URL')
                if origin_url:
                    repo_url = origin_url
                    homepage_url = origin_url
        
        # If still no version found, try master or main branch
        if not latest_version:
            for branch in ['master', 'main']:
                branch_url = f"https://proxy.golang.org/{module_path}/@v/{branch}.info"
                branch_response = session.get(branch_url, timeout=10)
                if branch_response.status_code == 200:
                    branch_data = branch_response.json()
                    latest_version = branch_data.get('Version')
                    if latest_version:
                        # Also try to get Origin URL from branch info
                        origin = branch_data.get('Origin', {})
                        if origin and repo_url == "nan":
                            origin_url = origin.get('URL')
                            if origin_url:
                                repo_url = origin_url
                                homepage_url = origin_url
                        break
        
        # If we have a version but still no repo URL, fetch the .info file
        if latest_version and repo_url == "nan":
            info_url = f"https://proxy.golang.org/{module_path}/@v/{latest_version}.info"
            info_response = session.get(info_url, timeout=10)
            if info_response.status_code == 200:
                info_data = info_response.json()
                origin = info_data.get('Origin', {})
                if origin:
                    origin_url = origin.get('URL')
                    if origin_url:
                        repo_url = origin_url
                        homepage_url = origin_url
        
        # No fallback inference - only use data from official Go proxy API
    except (requests.exceptions.RequestException, ValueError, KeyError, IndexError):
        # If API call fails, keep values as "nan" - do not infer from module path
        pass
    
    return homepage_url, repo_url

def mine_go_packages(output_dir=None, output_filename=None):
    """Mines Go packages to get the whole list from the Go module index."""
    
    # Download module list
    modules = download_go_index()
    
    if not modules:
        print("Failed to download Go modules or no modules found.")
        return
    
    print(f"Found {len(modules)} unique Go modules")
    
    # Create the path to the output file
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        output_dir = os.path.abspath(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if output_filename is None:
        output_filename = DEFAULT_OUTPUT_FILENAME
    elif not output_filename.endswith('.csv'):
        output_filename += '.csv'
    
    output_file = os.path.join(output_dir, output_filename)
    
    print("Fetching detailed information for each module...")
    print("Using parallel processing with 100 workers for faster execution...")
    print("This will take several hours due to the large number of modules (~2M)...")
    
    # Use parallel processing with many workers for speed
    results = {}
    session = create_session()
    
    def fetch_module_wrapper(idx_and_module):
        """Wrapper to fetch module info with its own session."""
        idx, module = idx_and_module
        module_path = module['path']
        # Create a session per worker for better connection pooling
        worker_session = create_session()
        try:
            homepage_url, repo_url = get_module_info(module_path, worker_session)
            return idx, module_path, homepage_url, repo_url
        finally:
            worker_session.close()
    
    with ThreadPoolExecutor(max_workers=100) as executor:
        # Submit all tasks
        future_to_module = {
            executor.submit(fetch_module_wrapper, (idx, module)): (idx, module)
            for idx, module in enumerate(modules, start=1)
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_module), total=len(future_to_module), desc="Processing modules"):
            idx, original_module = future_to_module[future]
            try:
                idx, module_path, homepage_url, repo_url = future.result()
                results[idx] = (module_path, homepage_url, repo_url)
            except Exception as e:
                # If something went wrong, store with nan values
                results[idx] = (original_module['path'], "nan", "nan")
    
    # Write results to CSV in order
    print("Writing results to CSV...")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Platform", "Name", "Homepage URL", "Repository URL"])
        
        for idx in sorted(results.keys()):
            module_path, homepage_url, repo_url = results[idx]
            writer.writerow([
                idx,
                "Go",
                module_path,
                homepage_url,
                repo_url,
            ])
    
    print(f"Successfully saved {len(modules)} Go modules to {output_file}")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Mine Go module packages from the official Go module index.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default output location
  python mine_go.py
  
  # Specify custom output directory
  python mine_go.py --output-dir /path/to/output
  
  # Specify custom filename
  python mine_go.py --output-file custom_go_modules.csv
  
  # Specify both directory and filename
  python mine_go.py --output-dir ./data --output-file go_packages.csv
        '''
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help=f'Output directory for the CSV file. Default: {DEFAULT_OUTPUT_DIR}'
    )
    
    parser.add_argument(
        '-f', '--output-file',
        type=str,
        default=None,
        help=f'Output filename. Default: {DEFAULT_OUTPUT_FILENAME} (.csv extension will be added if missing)'
    )
    
    args = parser.parse_args()
    
    mine_go_packages(output_dir=args.output_dir, output_filename=args.output_file)

if __name__ == "__main__":
    main()
