#!/usr/bin/env python3
"""
Re-mine Contributors Count
Re-mines all repositories' contributor counts using REST API with anon=true and updates if different.

Command to check contributor count using rest api:
curl -s "https://api.github.com/repos/OWNER/REPO/contributors?per_page=1&anon=true" -I | grep -i "^link:"
"""
import json
import requests
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "../../Resource/Dataset/"
DEFAULT_INPUT_FILE = DATASET_DIR / "Metric-Miner-Github/github_metrics.json"
DEFAULT_OUTPUT_FILE = DATASET_DIR / "Metric-Miner-Github/github_metrics.json"
DEFAULT_BACKUP_FILE = DATASET_DIR / "Metric-Miner-Github/github_metrics_backup.json"


# ============================================================================
# GITHUB API CONFIGURATION
# ============================================================================

# Default token (can be overridden via command line)
DEFAULT_GITHUB_TOKEN = ""
BASE_URL = "https://api.github.com"
MAX_RETRIES = 3
RETRY_DELAY = 2


# ============================================================================
# TOKEN ROTATION MANAGER
# ============================================================================

class TokenManager:
    """Manages multiple GitHub tokens with automatic rotation on rate limit."""
    
    def __init__(self, tokens: List[str]):
        """
        Initialize with a list of GitHub tokens.
        
        Args:
            tokens: List of GitHub personal access tokens
        """
        self.tokens = [t.strip() for t in tokens if t and t.strip()]
        if not self.tokens:
            raise ValueError("At least one GitHub token is required")
        
        self.current_index = 0
        self.token_status = {}  # token -> (remaining, reset_timestamp)
        
        print(f"✓ Initialized with {len(self.tokens)} token(s)")
    
    @property
    def current_token(self) -> str:
        """Get the current active token."""
        return self.tokens[self.current_index]
    
    def get_headers(self) -> dict:
        """Get headers with the current authentication token."""
        return {'Authorization': f'token {self.current_token}'}
    
    def check_token_rate_limit(self, token: str) -> Tuple[int, int]:
        """
        Check rate limit status for a specific token.
        
        Returns:
            Tuple of (remaining, reset_timestamp)
        """
        try:
            headers = {'Authorization': f'token {token}'}
            response = requests.get(f"{BASE_URL}/rate_limit", headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                core = data.get('resources', {}).get('core', {})
                remaining = core.get('remaining', 0)
                reset = core.get('reset', 0)
                return (remaining, reset)
        except Exception as e:
            print(f"  ⚠ Error checking rate limit: {e}")
        return (0, 0)
    
    def check_current_rate_limit(self) -> Tuple[int, int]:
        """Check rate limit for the current token."""
        remaining, reset = self.check_token_rate_limit(self.current_token)
        self.token_status[self.current_token] = (remaining, reset)
        return (remaining, reset)
    
    def rotate_token(self) -> bool:
        """
        Rotate to the next available token with remaining rate limit.
        
        Returns:
            True if successfully rotated to a token with remaining limit,
            False if all tokens are exhausted.
        """
        original_index = self.current_index
        
        # Try each token
        for _ in range(len(self.tokens)):
            self.current_index = (self.current_index + 1) % len(self.tokens)
            remaining, reset = self.check_token_rate_limit(self.tokens[self.current_index])
            self.token_status[self.tokens[self.current_index]] = (remaining, reset)
            
            if remaining > 10:
                if self.current_index != original_index:
                    print(f"  🔄 Rotated to token #{self.current_index + 1} ({remaining} requests remaining)")
                return True
        
        # All tokens exhausted
        return False
    
    def wait_for_rate_limit(self):
        """
        Ensure we have a token with available rate limit.
        Rotates tokens if needed, waits if all are exhausted.
        """
        remaining, reset = self.check_current_rate_limit()
        
        if remaining > 10:
            return
        
        # Try to rotate to another token
        if self.rotate_token():
            return
        
        # All tokens exhausted, find the earliest reset time
        earliest_reset = float('inf')
        for token in self.tokens:
            _, reset = self.token_status.get(token, (0, 0))
            if reset > 0:
                earliest_reset = min(earliest_reset, reset)
        
        if earliest_reset < float('inf'):
            wait_time = max(earliest_reset - time.time(), 0) + 5
            print(f"  ⏳ All tokens exhausted. Waiting {wait_time:.0f}s for reset...")
            time.sleep(wait_time)
            
            # After waiting, find a token with available limit
            self.rotate_token()
    
    def print_status(self):
        """Print the status of all tokens."""
        print("\nToken Status:")
        for i, token in enumerate(self.tokens):
            remaining, reset = self.check_token_rate_limit(token)
            self.token_status[token] = (remaining, reset)
            marker = "→" if i == self.current_index else " "
            print(f"  {marker} Token #{i + 1}: {remaining} requests remaining")


# Global token manager instance (initialized in main)
token_manager: TokenManager = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_headers() -> dict:
    """Get headers with the current authentication token."""
    global token_manager
    if token_manager:
        return token_manager.get_headers()
    return {'Authorization': f'token {DEFAULT_GITHUB_TOKEN}'}


def check_rate_limit() -> Tuple[int, int]:
    """
    Check current rate limit status.
    Returns: (remaining, reset_timestamp)
    """
    global token_manager
    if token_manager:
        return token_manager.check_current_rate_limit()
    
    try:
        response = requests.get(f"{BASE_URL}/rate_limit", headers=get_headers(), timeout=5)
        if response.status_code == 200:
            data = response.json()
            core = data.get('resources', {}).get('core', {})
            remaining = core.get('remaining', 0)
            reset = core.get('reset', 0)
            return (remaining, reset)
    except Exception as e:
        print(f"Error checking rate limit: {e}")
    return (0, 0)


def wait_for_rate_limit():
    """Wait for rate limit to reset if needed, with token rotation."""
    global token_manager
    if token_manager:
        token_manager.wait_for_rate_limit()
        return
    
    remaining, reset = check_rate_limit()
    if remaining < 10:
        wait_time = max(reset - time.time(), 0) + 5
        print(f"⏳ Rate limit low ({remaining} remaining). Waiting {wait_time:.0f}s...")
        time.sleep(wait_time)


def get_contributors_count_with_anon(owner: str, repo: str) -> int:
    """
    Get contributors count for a repository with anon=true parameter.
    Returns the correct count or -1 if failed.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            wait_for_rate_limit()
            
            contributors_url = f"{BASE_URL}/repos/{owner}/{repo}/contributors"
            params = {"per_page": 1, "anon": "true"}  # Include anonymous contributors
            
            response = requests.get(contributors_url, headers=get_headers(), params=params, timeout=10)
            
            if response.status_code == 200:
                # Try pagination header first (most reliable)
                if "Link" in response.headers:
                    link_header = response.headers["Link"]
                    match = re.search(r'page=(\d+)>; rel="last"', link_header)
                    if match:
                        count = int(match.group(1))
                        if count > 0:
                            return count
                
                # Fallback to counting array length
                contributors = response.json()
                if isinstance(contributors, list):
                    return len(contributors)
                else:
                    print(f"  ⚠ Invalid response format for {owner}/{repo}")
                    return -1
            
            elif response.status_code == 404:
                print(f"  ⚠ Repository {owner}/{repo} not found")
                return 0
            
            elif response.status_code == 403:
                # Rate limit or forbidden
                print(f"  ⚠ Forbidden (403) for {owner}/{repo}, retrying...")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY ** (attempt + 1))
                    continue
                return -1
            
            else:
                print(f"  ⚠ HTTP {response.status_code} for {owner}/{repo}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY ** (attempt + 1))
                    continue
                return -1
        
        except requests.exceptions.Timeout:
            print(f"  ⏱ Timeout for {owner}/{repo}, attempt {attempt + 1}/{MAX_RETRIES + 1}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY ** (attempt + 1))
                continue
            return -1
        
        except Exception as e:
            print(f"  ✗ Error fetching {owner}/{repo}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY ** (attempt + 1))
                continue
            return -1
    
    return -1


def load_metrics_data(input_file: Path) -> dict:
    """Load the GitHub metrics JSON file."""
    print(f"Loading metrics from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data.get('packages', {}))} packages")
    return data


def save_metrics_data(output_file: Path, data: dict):
    """Save the updated metrics data."""
    print(f"\nSaving updated metrics to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("✓ Saved successfully")


def backup_metrics_data(input_file: Path, backup_file: Path):
    """Create a backup of the original metrics file."""
    print(f"Creating backup: {backup_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("✓ Backup created")


def get_all_packages(data: dict) -> List[Tuple[str, dict]]:
    """
    Get all packages from the data.
    Returns list of (package_key, package_data) tuples.
    """
    packages = data.get('packages', {})
    return [(key, value) for key, value in packages.items()]


def remining_all_contributors(data: dict, packages_list: List[Tuple[str, dict]]) -> Tuple[int, int, int]:
    """
    Re-mine contributors count for all packages.
    Returns (updated_count, unchanged_count, failed_count).
    """
    updated_count = 0
    unchanged_count = 0
    failed_count = 0
    
    print(f"\n{'='*70}")
    print(f"Re-mining contributors for {len(packages_list)} packages")
    print(f"{'='*70}\n")
    
    for package_key, package_data in tqdm(packages_list, desc="Re-mining contributors"):
        owner_repo = package_data.get('owner_repo', '')
        
        if '/' not in owner_repo:
            print(f"  ✗ Invalid owner_repo format: {owner_repo}")
            failed_count += 1
            continue
        
        owner, repo = owner_repo.split('/', 1)
        
        # Get the correct contributor count with anon=true
        new_count = get_contributors_count_with_anon(owner, repo)
        
        if new_count >= 0:
            old_count = package_data.get('contributors', 0)
            
            if new_count != old_count:
                data['packages'][package_key]['contributors'] = new_count
                tqdm.write(f"  ✓ {owner_repo}: {old_count} → {new_count}")
                updated_count += 1
            else:
                unchanged_count += 1
        else:
            tqdm.write(f"  ✗ Failed to fetch: {owner_repo}")
            failed_count += 1
    
    return updated_count, unchanged_count, failed_count


def main():
    """Main function."""
    global token_manager
    
    parser = argparse.ArgumentParser(
        description='Re-mine and update all contributor counts in GitHub metrics data'
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help='Input JSON file path (default: github_metrics.json)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help='Output JSON file path (default: same as input)'
    )
    parser.add_argument(
        '-b', '--backup',
        type=Path,
        default=DEFAULT_BACKUP_FILE,
        help='Backup file path (default: github_metrics_backup.json)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup file'
    )
    parser.add_argument(
        '-t', '--tokens',
        type=str,
        nargs='+',
        help='GitHub API tokens (space-separated). Multiple tokens enable rotation on rate limit.'
    )
    parser.add_argument(
        '--token-file',
        type=Path,
        help='File containing GitHub tokens (one per line)'
    )
    
    args = parser.parse_args()
    
    # Collect tokens from all sources
    tokens = []
    
    # Load tokens from command line
    if args.tokens:
        tokens.extend(args.tokens)
    
    # Load tokens from file if specified
    if args.token_file and args.token_file.exists():
        print(f"Loading tokens from: {args.token_file}")
        with open(args.token_file, 'r') as f:
            file_tokens = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            tokens.extend(file_tokens)
            print(f"  Loaded {len(file_tokens)} token(s) from file")
    
    # Use default token if no tokens provided
    if not tokens:
        tokens = [DEFAULT_GITHUB_TOKEN]
    
    # Remove duplicates while preserving order
    seen_tokens = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen_tokens:
            seen_tokens.add(t)
            unique_tokens.append(t)
    tokens = unique_tokens
    
    # Initialize token manager
    try:
        token_manager = TokenManager(tokens)
    except ValueError as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Validate input file exists
    if not args.input.exists():
        print(f"✗ Error: Input file not found: {args.input}")
        return 1
    
    # Check rate limit status for all tokens
    print("\nChecking GitHub API rate limits...")
    token_manager.print_status()
    
    # Load data
    data = load_metrics_data(args.input)
    
    # Get all packages
    print(f"\nPreparing to re-mine all packages...")
    all_packages = get_all_packages(data)
    
    if not all_packages:
        print(f"✓ No packages found in the dataset")
        return 0
    
    print(f"Found {len(all_packages)} packages to process")
    
    # Create backup
    if not args.no_backup:
        backup_metrics_data(args.input, args.backup)
    
    # Re-mine all contributors
    updated_count, unchanged_count, failed_count = remining_all_contributors(data, all_packages)
    
    # Save updated data if there were changes
    if updated_count > 0:
        save_metrics_data(args.output, data)
    else:
        print("\n✓ No changes detected, skipping save")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Total packages processed: {len(all_packages)}")
    print(f"  Updated (different count): {updated_count}")
    print(f"  Unchanged (same count): {unchanged_count}")
    print(f"  Failed: {failed_count}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    exit(main())
