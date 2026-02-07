#!/usr/bin/env python3
"""
Mine Unique Merged Contributors for Multirepo Packages

This script:
1. Parses github_metrics.json and filters for multirepo packages only
2. Groups multirepo packages by normalized name (after removing ecosystem patterns)
3. For each group, fetches all contributors from each repo
4. Calculates unique merged contributors across all repos in the group
5. Adds a "contributors_unique_merged" field to each entry

Example:
- microsoft/durabletask-java and microsoft/durabletask-js are grouped together
  because they both normalize to "microsoft/durabletask-"
- If java has [John, Mary, May] and js has [John, Mary, Martin]
- The unique merged count is 4 (John, Mary, May, Martin)

Usage:
    python mine_unique_contributors.py
    python mine_unique_contributors.py -i input.json -o output.json
    python mine_unique_contributors.py --dry-run  # Preview without saving
    
    # Multiple tokens for rate limit rotation
    python mine_unique_contributors.py -t token1 token2 token3
    python mine_unique_contributors.py --token-file tokens.txt
"""
import json
import requests
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "../../Resource/Dataset/"
DEFAULT_INPUT_FILE = DATASET_DIR / "Metric-Miner-Github/github_metrics.json"
DEFAULT_OUTPUT_FILE = DATASET_DIR / "Metric-Miner-Github/github_metrics.json"
DEFAULT_BACKUP_FILE = DATASET_DIR / "Metric-Miner-Github/github_metrics_backup_unique.json"


# ============================================================================
# GITHUB API CONFIGURATION
# ============================================================================

# Default tokens (can be overridden via command line)
DEFAULT_GITHUB_TOKENS = []
BASE_URL = "https://api.github.com"
MAX_RETRIES = 3
RETRY_DELAY = 2
CONTRIBUTORS_PER_PAGE = 100  # Max allowed by GitHub API


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
# ECOSYSTEM SUFFIX PATTERNS (Same as filter_multirepo_common_package.py)
# ============================================================================

ECOSYSTEM_SUFFIX_PATTERNS = {
    'PyPI': [
        'python', 'py', 'pypi', 'python2', 'python3', 'py2', 'py3',
        'cpython', 'pysdk', 'pyclient', 'pylib'
    ],
    'Crates': [
        'rust', 'rs', 'cargo', 'rustlang', 'crate', 'crates'
    ],
    'Go': [
        'go', 'golang', 'goclient', 'gosdk', 'golib'
    ],
    'NPM': [
        'js', 'javascript', 'node', 'nodejs', 'npm', 'ts', 'typescript',
        'jsclient', 'tsclient', 'jssdk', 'tssdk', 'jslib', 'tslib'
    ],
    'Maven': [
        'java', 'jvm', 'maven', 'scala', 'kotlin', 'kt', 'kts',
        'javaclient', 'javasdk', 'javalib', 'scalaclient', 'scalalib'
    ],
    'Ruby': [
        'ruby', 'rb', 'gem', 'rubygem', 'rubyclient', 'rubylib'
    ],
    'PHP': [
        'php', 'php5', 'php7', 'php8', 'phpclient', 'phpsdk', 'phplib'
    ],
    'Other': [
        'net', 'dotnet', 'csharp', 'cs', 'fsharp', 'fs',
        'cpp', 'cxx', 'cplusplus', 'c', 'clang',
        'swift', 'swiftclient',
        'elixir', 'ex', 'exs',
        'dart', 'flutter',
        'perl', 'pl',
        'lua',
        'r', 'rlang',
        'haskell', 'hs',
        'ocaml', 'ml',
        'erlang', 'erl',
        'clojure', 'clj',
        'groovy', 'gvy'
    ]
}

# Flatten all suffixes into a single list
ECOSYSTEM_SUFFIXES = []
for ecosystem, suffixes in ECOSYSTEM_SUFFIX_PATTERNS.items():
    ECOSYSTEM_SUFFIXES.extend(suffixes)

# Remove duplicates while preserving order
seen = set()
ECOSYSTEM_SUFFIXES = [x for x in ECOSYSTEM_SUFFIXES if not (x in seen or seen.add(x))]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_headers() -> dict:
    """Get headers with the current authentication token."""
    global token_manager
    if token_manager:
        return token_manager.get_headers()
    return {'Authorization': f'token {DEFAULT_GITHUB_TOKENS[0]}'}


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


def remove_ecosystem_patterns(repo_name: str) -> str:
    """
    Remove ecosystem patterns from a repository name, keeping separators intact.
    
    Examples:
        - 'libsql-js' -> 'libsql-'
        - 'libsql-python' -> 'libsql-'
        - 'durabletask-java' -> 'durabletask-'
    
    Args:
        repo_name: Repository name (last part of owner/repo)
    
    Returns:
        Name with ecosystem patterns removed but separators preserved
    """
    if not repo_name:
        return None
    
    normalized = repo_name.lower()
    
    # Remove all ecosystem patterns, but keep separators intact
    for suffix in ECOSYSTEM_SUFFIXES:
        # Remove pattern with any common separator or at boundaries
        # This handles: -suffix, _suffix, .suffix, suffix-, suffix_, suffix.
        pattern = r'(?:^|(?<=[-_.\s]))' + re.escape(suffix) + r'(?=[-_.\s]|$)'
        normalized = re.sub(pattern, '', normalized)
    
    # Return None if nothing meaningful remains
    if not normalized or len(normalized) < 2:
        return None
    
    return normalized


def get_normalized_key(owner_repo: str) -> str:
    """
    Get normalized key for grouping multirepo packages.
    
    Args:
        owner_repo: Full owner/repo string (e.g., "microsoft/durabletask-java")
    
    Returns:
        Normalized key (e.g., "microsoft/durabletask-") or None if invalid
    """
    if '/' not in owner_repo:
        return None
    
    owner, repo = owner_repo.split('/', 1)
    normalized_repo = remove_ecosystem_patterns(repo)
    
    if not normalized_repo:
        return None
    
    return f"{owner.lower()}/{normalized_repo}"


def get_all_contributors(owner: str, repo: str) -> Set[str]:
    """
    Get all contributors for a repository using pagination.
    Returns a set of contributor identifiers (login for users, email for anonymous).
    
    Args:
        owner: Repository owner
        repo: Repository name
    
    Returns:
        Set of contributor identifiers
    """
    contributors = set()
    page = 1
    
    while True:
        for attempt in range(MAX_RETRIES + 1):
            try:
                wait_for_rate_limit()
                
                url = f"{BASE_URL}/repos/{owner}/{repo}/contributors"
                params = {
                    "per_page": CONTRIBUTORS_PER_PAGE,
                    "anon": "true",
                    "page": page
                }
                
                response = requests.get(url, headers=get_headers(), params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data:  # Empty page, we're done
                        return contributors
                    
                    for contributor in data:
                        if contributor.get('type') == 'Anonymous':
                            # For anonymous contributors, use email as identifier
                            email = contributor.get('email', '')
                            if email:
                                contributors.add(f"anon:{email.lower()}")
                        else:
                            # For regular users, use login
                            login = contributor.get('login', '')
                            if login:
                                contributors.add(login.lower())
                    
                    # Check if there are more pages
                    if len(data) < CONTRIBUTORS_PER_PAGE:
                        return contributors
                    
                    page += 1
                    break  # Success, continue to next page
                
                elif response.status_code == 404:
                    # Repository not found
                    return contributors
                
                elif response.status_code == 403:
                    # Rate limit or forbidden
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY ** (attempt + 1))
                        continue
                    return contributors
                
                else:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY ** (attempt + 1))
                        continue
                    return contributors
            
            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY ** (attempt + 1))
                    continue
                return contributors
            
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY ** (attempt + 1))
                    continue
                return contributors
        else:
            # All retries exhausted for this page
            return contributors
    
    return contributors


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

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


def filter_multirepo_packages(data: dict) -> Dict[str, dict]:
    """
    Filter packages to only include multirepo entries.
    
    Returns:
        Dictionary of package_key -> package_data for multirepo packages
    """
    multirepo_packages = {}
    packages = data.get('packages', {})
    
    for package_key, package_data in packages.items():
        source = package_data.get('source', '')
        if source == 'multirepo':
            multirepo_packages[package_key] = package_data
    
    return multirepo_packages


def group_multirepo_by_normalized_name(multirepo_packages: Dict[str, dict]) -> Dict[str, List[Tuple[str, dict]]]:
    """
    Group multirepo packages by their normalized name (after removing ecosystem patterns).
    
    Args:
        multirepo_packages: Dictionary of package_key -> package_data
    
    Returns:
        Dictionary of normalized_key -> list of (package_key, package_data) tuples
    """
    groups = defaultdict(list)
    ungrouped_count = 0
    
    for package_key, package_data in multirepo_packages.items():
        owner_repo = package_data.get('owner_repo', '')
        normalized_key = get_normalized_key(owner_repo)
        
        if normalized_key:
            groups[normalized_key].append((package_key, package_data))
        else:
            ungrouped_count += 1
    
    if ungrouped_count > 0:
        print(f"  ⚠ {ungrouped_count} packages could not be normalized")
    
    return groups


def process_multirepo_groups(data: dict, groups: Dict[str, List[Tuple[str, dict]]], dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Process each multirepo group to calculate unique merged contributors.
    
    Args:
        data: Full metrics data (will be modified in place)
        groups: Dictionary of normalized_key -> list of (package_key, package_data)
        dry_run: If True, don't modify data, just show what would be done
    
    Returns:
        Tuple of (groups_processed, packages_updated, groups_failed)
    """
    groups_processed = 0
    packages_updated = 0
    groups_failed = 0
    
    print(f"\n{'='*70}")
    print(f"Processing {len(groups)} multirepo groups")
    print(f"{'='*70}\n")
    
    for normalized_key, packages_in_group in tqdm(groups.items(), desc="Processing groups"):
        # Collect all unique contributors across all repos in this group
        all_contributors = set()
        group_success = True
        
        for package_key, package_data in packages_in_group:
            owner_repo = package_data.get('owner_repo', '')
            
            if '/' not in owner_repo:
                continue
            
            owner, repo = owner_repo.split('/', 1)
            
            # Get all contributors for this repo
            contributors = get_all_contributors(owner, repo)
            
            if contributors:
                all_contributors.update(contributors)
            else:
                # If we couldn't get contributors for any repo, mark as partial failure
                tqdm.write(f"  ⚠ Could not fetch contributors for {owner_repo}")
        
        # Calculate unique merged count
        unique_merged_count = len(all_contributors)
        
        if unique_merged_count > 0:
            groups_processed += 1
            
            # Update all packages in this group with the merged count
            for package_key, package_data in packages_in_group:
                old_count = package_data.get('contributors', 0)
                
                if not dry_run:
                    data['packages'][package_key]['contributors_unique_merged'] = unique_merged_count
                
                packages_updated += 1
                
                # Log significant changes
                if len(packages_in_group) > 1:
                    tqdm.write(f"  ✓ {normalized_key}: {len(packages_in_group)} repos → {unique_merged_count} unique contributors")
        else:
            groups_failed += 1
            tqdm.write(f"  ✗ Failed to process group: {normalized_key}")
    
    return groups_processed, packages_updated, groups_failed


def main():
    """Main function."""
    global token_manager
    
    parser = argparse.ArgumentParser(
        description='Mine unique merged contributors for multirepo packages'
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
        help='Backup file path (default: github_metrics_backup_unique.json)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without saving'
    )
    parser.add_argument(
        '-t', '--tokens',
        type=str,
        nargs='+',
        default=DEFAULT_GITHUB_TOKENS,
        help='GitHub API tokens (space-separated). Multiple tokens enable rotation on rate limit.'
    )
    parser.add_argument(
        '--token-file',
        type=Path,
        help='File containing GitHub tokens (one per line)'
    )
    
    args = parser.parse_args()
    
    # Collect tokens from all sources
    tokens = list(args.tokens) if args.tokens else []
    
    # Load tokens from file if specified
    if args.token_file and args.token_file.exists():
        print(f"Loading tokens from: {args.token_file}")
        with open(args.token_file, 'r') as f:
            file_tokens = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            tokens.extend(file_tokens)
            print(f"  Loaded {len(file_tokens)} token(s) from file")
    
    # Remove duplicates while preserving order
    seen_tokens = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen_tokens:
            seen_tokens.add(t)
            unique_tokens.append(t)
    tokens = unique_tokens
    
    if not tokens:
        print("✗ Error: No GitHub tokens provided")
        return 1
    
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
    
    # Filter for multirepo packages only
    print("\nFiltering for multirepo packages...")
    multirepo_packages = filter_multirepo_packages(data)
    print(f"  Found {len(multirepo_packages)} multirepo packages")
    
    if not multirepo_packages:
        print("✓ No multirepo packages found")
        return 0
    
    # Group by normalized name
    print("\nGrouping by normalized name (after removing ecosystem patterns)...")
    groups = group_multirepo_by_normalized_name(multirepo_packages)
    
    # Filter to only groups with 2+ packages (actual multirepo sets)
    actual_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    single_packages = {k: v for k, v in groups.items() if len(v) == 1}
    
    print(f"  Groups with 2+ packages: {len(actual_groups)}")
    print(f"  Single packages (no match): {len(single_packages)}")
    
    # Show some examples
    if actual_groups:
        print("\nExamples of multirepo groups:")
        for i, (normalized_key, packages_in_group) in enumerate(list(actual_groups.items())[:5]):
            repos = [p[1].get('owner_repo', '') for p in packages_in_group]
            print(f"  {i+1}. {normalized_key}")
            for repo in repos[:3]:
                print(f"      - {repo}")
            if len(repos) > 3:
                print(f"      ... and {len(repos) - 3} more")
        if len(actual_groups) > 5:
            print(f"  ... and {len(actual_groups) - 5} more groups")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be saved]")
    
    # Create backup
    if not args.no_backup and not args.dry_run:
        backup_metrics_data(args.input, args.backup)
    
    # Process groups and calculate unique merged contributors
    groups_processed, packages_updated, groups_failed = process_multirepo_groups(
        data, actual_groups, dry_run=args.dry_run
    )
    
    # Also process single packages (set their unique merged count to their own count)
    print("\nProcessing single-repo packages...")
    for normalized_key, packages_in_group in tqdm(single_packages.items(), desc="Single packages"):
        for package_key, package_data in packages_in_group:
            # For single packages, unique merged count equals null
            if not args.dry_run:
                data['packages'][package_key]['contributors_unique_merged'] = ''
            packages_updated += 1
    
    # Save updated data
    if not args.dry_run:
        save_metrics_data(args.output, data)
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Total multirepo packages: {len(multirepo_packages)}")
    print(f"  Multirepo groups (2+ packages): {len(actual_groups)}")
    print(f"  Single packages: {len(single_packages)}")
    print(f"  Groups processed successfully: {groups_processed}")
    print(f"  Packages updated: {packages_updated}")
    print(f"  Groups failed: {groups_failed}")
    if args.dry_run:
        print("  [DRY RUN - No changes saved]")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    exit(main())
