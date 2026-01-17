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

# Checkpoint files
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.checkpoint'))
MODULE_NAMES_FILE = os.path.join(CHECKPOINT_DIR, "module_names.txt")  # Store names line by line
INDEX_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "index_checkpoint.json")

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

def append_module_names(module_names):
    """Append module names to file (memory efficient)."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(MODULE_NAMES_FILE, 'a', encoding='utf-8') as f:
        for name in module_names:
            f.write(name + '\n')

def get_module_names_count():
    """Get total count of module names without loading all into memory."""
    if not os.path.exists(MODULE_NAMES_FILE):
        return 0
    count = 0
    with open(MODULE_NAMES_FILE, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count

def load_module_names_batch(start_idx, batch_size=1000):
    """Load a batch of module names from file."""
    if not os.path.exists(MODULE_NAMES_FILE):
        return []
    
    modules = []
    with open(MODULE_NAMES_FILE, 'r', encoding='utf-8') as f:
        # Skip to start index
        for _ in range(start_idx):
            next(f, None)
        
        # Read batch
        for i, line in enumerate(f):
            if i >= batch_size:
                break
            modules.append(line.strip())
    
    return modules

def save_index_checkpoint(modules_set, since, batch_count, total_entries):
    """Save index download checkpoint to disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Append new modules to file
    if isinstance(modules_set, set):
        # On first call or when we have a set, we need to check what's already saved
        existing_count = get_module_names_count()
        if existing_count == 0:
            # First time, save all
            append_module_names(list(modules_set))
        # If existing_count > 0, new modules should already be appended
    
    total_modules = get_module_names_count()
    
    checkpoint_data = {
        'modules_count': total_modules,
        'since': since,
        'batch_count': batch_count,
        'total_entries': total_entries,
        'timestamp': time.time()
    }
    with open(INDEX_CHECKPOINT, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f)

def load_index_checkpoint():
    """Load index download checkpoint from disk."""
    if os.path.exists(INDEX_CHECKPOINT):
        try:
            with open(INDEX_CHECKPOINT, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Load module names as a set from file for deduplication
                modules_set = set()
                if os.path.exists(MODULE_NAMES_FILE):
                    with open(MODULE_NAMES_FILE, 'r', encoding='utf-8') as mf:
                        for line in mf:
                            modules_set.add(line.strip())
                
                print(f"Resuming from checkpoint: batch {data['batch_count']}, {len(modules_set)} modules")
                return (
                    modules_set,
                    data['since'],
                    data['batch_count'],
                    data['total_entries']
                )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Starting fresh")
            return set(), "", 0, 0
    return set(), "", 0, 0

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
    checkpoint_interval = 100  # Save more frequently for safety
    
    # Try to load from checkpoint
    print("Checking for existing checkpoint...")
    modules_set, since, batch_count, total_entries = load_index_checkpoint()
    
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
            new_modules_list = []
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
                        new_modules_list.append(module_path)
                    
                    # Always update timestamp to the last entry's timestamp
                    ts = entry.get('Timestamp')
                    if ts:
                        last_timestamp = ts
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Append new modules to file immediately to save memory
            if new_modules_list:
                append_module_names(new_modules_list)
            
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
                save_index_checkpoint(None, since, batch_count, total_entries)  # Pass None since already appended
            
            # Update progress bar - set to current batch count and update postfix
            pbar.update(1)
            pbar.set_postfix(unique=f"{len(modules_set):,}", new=len(new_modules_list), total=f"{total_entries:,}")
            
            # Clear modules_set periodically to save memory (we have them in file)
            if len(modules_set) > 100000:
                modules_set.clear()
                # Reload from file for deduplication
                with open(MODULE_NAMES_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        modules_set.add(line.strip())
            
            # If we got less than 2000 entries, we're likely at the end
            # Note: The API returns up to 2000 entries per request
            if entries_processed < 2000:
                pbar.write(f"\nReceived {entries_processed} entries (less than 2000), reached end of index")
                break
        
        pbar.close()
        
        print(f"\nFetched {batch_count} batches, {total_entries:,} total entries")
        
        # Save final checkpoint (DO NOT DELETE - will be cleaned up after full processing)
        save_index_checkpoint(None, since, batch_count, total_entries)
        
    except KeyboardInterrupt:
        if 'pbar' in locals():
            pbar.close()
        print(f"\n\nInterrupted! Progress saved to checkpoint.")
        print(f"Run the script again to resume from batch {batch_count}")
        # Save final checkpoint
        save_index_checkpoint(None, since, batch_count, total_entries)
        raise
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading Go module index: {e}")
        return []
    finally:
        session.close()
    
    # Return module count instead of loading all into memory
    total_modules = get_module_names_count()
    return total_modules

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

def load_processed_modules(output_file):
    """Load already processed modules from the CSV file."""
    processed = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row['ID'])
                    processed[idx] = (row['Name'], row['Homepage URL'], row['Repository URL'])
            print(f"Found {len(processed)} already processed modules")
        except (csv.Error, KeyError, ValueError) as e:
            print(f"Warning: Failed to load existing CSV: {e}")
            return {}
    return processed

def save_results_incrementally(output_file, results, write_header=False):
    """Save results to CSV incrementally."""
    mode = 'w' if write_header else 'a'
    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
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
        f.flush()  # Ensure data is written to disk
        os.fsync(f.fileno())  # Force write to disk

def mine_go_packages(output_dir=None, output_filename=None):
    """Mines Go packages to get the whole list from the Go module index."""
    
    # Download module list
    total_modules = download_go_index()
    
    if total_modules == 0:
        print("Failed to download Go modules or no modules found.")
        return
    
    print(f"Found {total_modules:,} unique Go modules")
    
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
    
    # Load already processed modules
    print("Checking for existing results...")
    processed_results = load_processed_modules(output_file)
    
    # Determine which modules still need processing
    total_to_process = total_modules - len(processed_results)
    
    if total_to_process <= 0:
        print("All modules already processed!")
        print(f"Results are in {output_file}")
        # Clean up checkpoint since everything is complete
        print("Cleaning up checkpoint files...")
        if os.path.exists(MODULE_NAMES_FILE):
            os.remove(MODULE_NAMES_FILE)
        if os.path.exists(INDEX_CHECKPOINT):
            os.remove(INDEX_CHECKPOINT)
        print("Done!")
        return
    
    print(f"Total modules to process: {total_to_process:,}")
    print(f"Already processed: {len(processed_results):,} modules")
    print("Using parallel processing with 20 workers to avoid overwhelming the server...")
    print("Processing in batches of 1000 modules to conserve memory...")
    print("This will take several hours due to the large number of modules...")
    
    # Use parallel processing with many workers for speed
    batch_size = 1000
    start_idx = 0
    processed_count = 0
    
    # If starting fresh, write header; otherwise append
    write_header = len(processed_results) == 0
    session = create_session()
    
    def fetch_module_wrapper(idx_and_module_path):
        """Wrapper to fetch module info with its own session."""
        idx, module_path = idx_and_module_path
        # Create a session per worker for better connection pooling
        worker_session = create_session()
        try:
            # Add small delay to avoid overwhelming the server
            time.sleep(0.02)  # 20ms delay
            homepage_url, repo_url = get_module_info(module_path, worker_session)
            return idx, module_path, homepage_url, repo_url
        finally:
            worker_session.close()
    
    try:
        while start_idx < total_modules:
            # Load batch of module names
            batch_modules = load_module_names_batch(start_idx, batch_size)
            if not batch_modules:
                break
            
            # Create list of modules to process in this batch
            modules_to_process_batch = []
            for i, module_path in enumerate(batch_modules):
                idx = start_idx + i + 1
                if idx not in processed_results:
                    modules_to_process_batch.append((idx, module_path))
            
            if not modules_to_process_batch:
                start_idx += batch_size
                continue
            
            print(f"\nProcessing batch {start_idx // batch_size + 1}: modules {start_idx + 1} to {start_idx + len(batch_modules)}")
            print(f"  {len(modules_to_process_batch)} modules to process in this batch")
            
            new_results = {}
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                # Submit all tasks for modules in this batch
                future_to_module = {
                    executor.submit(fetch_module_wrapper, (idx, module_path)): (idx, module_path)
                    for idx, module_path in modules_to_process_batch
                }
                
                # Process completed tasks with progress bar
                for future in tqdm(as_completed(future_to_module), total=len(future_to_module), desc=f"Batch {start_idx // batch_size + 1}"):
                    idx, original_path = future_to_module[future]
                    try:
                        idx, module_path, homepage_url, repo_url = future.result()
                        new_results[idx] = (module_path, homepage_url, repo_url)
                    except Exception as e:
                        # If something went wrong, store with nan values
                        new_results[idx] = (original_path, "nan", "nan")
                    
                    processed_count += 1
            
            # Save batch results to CSV immediately
            if new_results:
                print(f"  Saving {len(new_results)} modules to CSV...")
                save_results_incrementally(output_file, new_results, write_header)
                # Verify the save by checking file exists and size
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    print(f"  ✓ Saved to {output_file} (size: {file_size:,} bytes)")
                else:
                    print(f"  ✗ Warning: Output file not found after save!")
                write_header = False  # Don't write header again
            
            # Move to next batch
            start_idx += batch_size
        
        print("\nAll processing complete!")
        print(f"Total modules processed: {processed_count:,}")
        print(f"Successfully saved to {output_file}")
        
        # Clean up checkpoint files only after output is completely written
        print("Cleaning up checkpoint files...")
        if os.path.exists(MODULE_NAMES_FILE):
            os.remove(MODULE_NAMES_FILE)
        if os.path.exists(INDEX_CHECKPOINT):
            os.remove(INDEX_CHECKPOINT)
        print("Done!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        print("Progress has been saved. Run the script again to resume.")
        raise
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        print("Progress has been saved. Run the script again to resume.")
        raise

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
