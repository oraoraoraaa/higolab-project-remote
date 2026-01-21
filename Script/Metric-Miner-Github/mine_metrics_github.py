#!/usr/bin/env python3
"""
GitHub Metrics Miner
Mines GitHub repository metrics from Config-Locator results.
Outputs CSV files with stars, commits, PRs, issues, contributors, and language proportions.

Features:
- Parallel processing with configurable workers
- Checkpoint/resume support for interruption recovery
- Multi-token support with automatic rotation
- Language proportion analysis
"""
import os
import csv
import argparse
import requests
import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "../../Resource/Dataset/"
DEFAULT_INPUT_DIR = DATASET_DIR / "Directory-Structure-Miner"
DEFAULT_OUTPUT_DIR = DATASET_DIR / "Metric-Miner-Github"
DEFAULT_OUTPUT_FILE = "github_metrics.csv"
CHECKPOINT_DIR = SCRIPT_DIR / ".checkpoint"
LOG_FILE = SCRIPT_DIR / "processing.log"

# ============================================================================
# PARALLEL PROCESSING CONFIGURATION
# ============================================================================

MAX_WORKERS = 10  # Number of concurrent threads for API calls
MAX_RETRIES = 3   # Maximum number of retry attempts for network errors
RETRY_DELAY = 2   # Initial delay in seconds before retry (exponential backoff)

# ============================================================================
# THREAD-SAFE UTILITIES
# ============================================================================

print_lock = Lock()
log_lock = Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        tqdm.write(*args, **kwargs)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def log_message(logger, level, message):
    """Thread-safe logging function."""
    with log_lock:
        if level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'debug':
            logger.debug(message)


# ============================================================================
# TOKEN MANAGER (from detect_multirepo.py)
# ============================================================================

class TokenManager:
    """Manages GitHub API tokens with rotation and validation."""
    
    def __init__(self, tokens: List[str] = None):
        """Initialize token manager with a list of tokens."""
        self.tokens = tokens or []
        self.current_token_index = 0
        self.base_url = "https://api.github.com"
        self.lock = Lock()  # Thread-safe token rotation
        self.rate_limit_threshold = 10  # Start looking for alternatives when below this
        self.logger = None  # Will be set by main
    
    @property
    def current_token(self) -> Optional[str]:
        """Get the currently active token."""
        if not self.tokens:
            return None
        return self.tokens[self.current_token_index]
    
    @property
    def headers(self) -> dict:
        """Get headers with current authentication token."""
        headers = {}
        token = self.current_token
        if token:
            headers['Authorization'] = f'token {token}'
        return headers
    
    def rotate_token(self) -> bool:
        """Switch to the next available token. Returns True if switched, False if no other tokens."""
        with self.lock:
            if not self.tokens or len(self.tokens) <= 1:
                return False
            
            self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
            safe_print(f"  ⟳ Switched to token #{self.current_token_index + 1}")
            return True
    
    def get_next_token(self) -> Optional[str]:
        """Get next token in round-robin fashion for parallel requests."""
        with self.lock:
            if not self.tokens:
                return None
            token = self.tokens[self.current_token_index]
            self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
            return token
    
    def validate_tokens(self) -> None:
        """Check and display status of all provided tokens."""
        if not self.tokens:
            return
        
        print("\nToken Status Check:")
        print("-" * 65)
        print(f"{'Token':<12} {'Status':<15} {'Remaining':<15} {'Reset Time':<15}")
        print("-" * 65)
        
        for i, token in enumerate(self.tokens):
            headers = {'Authorization': f'token {token}'}
            try:
                response = requests.get(
                    f"{self.base_url}/rate_limit", headers=headers, timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    core = data.get('resources', {}).get('core', {})
                    remaining = core.get('remaining', 'N/A')
                    reset_timestamp = core.get('reset', 0)
                    reset_time = datetime.fromtimestamp(reset_timestamp).strftime('%H:%M:%S')
                    status = "✓ Valid" if remaining > 0 else "⚠ Depleted"
                elif response.status_code == 401:
                    status = "✗ Invalid"
                    remaining = "N/A"
                    reset_time = "N/A"
                else:
                    status = f"Error {response.status_code}"
                    remaining = "N/A"
                    reset_time = "N/A"
            
            except Exception:
                status = "Connection Error"
                remaining = "N/A"
                reset_time = "N/A"
            
            print(f"Token #{i+1:<6} {status:<15} {remaining:<15} {reset_time:<15}")
        
        print("-" * 65 + "\n")
    
    def get_rate_limit_status(self, token: str = None) -> Tuple[int, int]:
        """
        Get current rate limit status for a token.
        Returns: (remaining, reset_timestamp)
        """
        if token is None:
            token = self.current_token
        
        if not token:
            return (0, 0)
        
        headers = {'Authorization': f'token {token}'}
        try:
            response = requests.get(
                f"{self.base_url}/rate_limit", 
                headers=headers, 
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                core = data.get('resources', {}).get('core', {})
                remaining = core.get('remaining', 0)
                reset = core.get('reset', 0)
                return (remaining, reset)
        except Exception:
            pass
        return (0, 0)
    
    def find_available_token(self) -> Optional[str]:
        """
        Find a token with available rate limit.
        Returns token if found, None otherwise.
        """
        if not self.tokens:
            return None
        
        # Check all tokens for available rate limit
        for i in range(len(self.tokens)):
            token = self.tokens[i]
            remaining, reset = self.get_rate_limit_status(token)
            
            if remaining > self.rate_limit_threshold:
                # Found a good token, switch to it
                with self.lock:
                    self.current_token_index = i
                return token
        
        return None
    
    def wait_for_rate_limit_reset(self):
        """Wait for the rate limit to reset on any available token."""
        if not self.tokens:
            # No tokens, wait default time
            time.sleep(60)
            return
        
        # Find the token with the earliest reset time
        earliest_reset = float('inf')
        for token in self.tokens:
            remaining, reset = self.get_rate_limit_status(token)
            if reset > 0 and reset < earliest_reset:
                earliest_reset = reset
        
        if earliest_reset != float('inf'):
            wait_time = max(earliest_reset - time.time(), 0) + 5
            reset_time_str = datetime.fromtimestamp(earliest_reset).strftime('%H:%M:%S')
            safe_print(f"\n⏳ All tokens exhausted. Waiting {wait_time:.0f}s until {reset_time_str}...")
            if self.logger:
                log_message(self.logger, 'info', 
                           f"All tokens exhausted. Waiting {wait_time:.0f}s for rate limit reset at {reset_time_str}")
            time.sleep(wait_time)
        else:
            # Fallback: wait 60 seconds
            safe_print(f"\n⏳ Rate limit status unknown. Waiting 60 seconds...")
            time.sleep(60)
    
    def ensure_rate_limit(self) -> str:
        """
        Ensure we have rate limit available. 
        Will rotate tokens or wait as needed.
        Returns a token with available rate limit.
        """
        # Check current token
        if self.current_token:
            remaining, reset = self.get_rate_limit_status(self.current_token)
            
            if remaining > self.rate_limit_threshold:
                return self.current_token
            
            # Current token is low, try to find another
            available_token = self.find_available_token()
            if available_token:
                return available_token
            
            # All tokens exhausted, wait for reset
            self.wait_for_rate_limit_reset()
            
            # After waiting, return current token
            return self.current_token
        
        return None
    
    def check_rate_limit_from_response(self, response: requests.Response) -> bool:
        """Check rate limit from response headers and rotate token if needed. Returns True if rotated."""
        remaining = response.headers.get('X-RateLimit-Remaining')
        if remaining is not None and int(remaining) < self.rate_limit_threshold:
            return self.rotate_token()
        return False


# ============================================================================
# CHECKPOINT MANAGEMENT (inspired by mine_npm.py)
# ============================================================================

def get_checkpoint_file(txt_filename: str) -> Path:
    """Get checkpoint file path for a specific txt file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{txt_filename.replace('.txt', '')}_checkpoint.json"


def save_checkpoint(txt_filename: str, processed_packages: Dict[str, Dict], last_index: int):
    """Save checkpoint to disk."""
    checkpoint_file = get_checkpoint_file(txt_filename)
    checkpoint_data = {
        'last_index': last_index,
        'processed_count': len(processed_packages),
        'timestamp': time.time(),
        'packages': processed_packages
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)


def load_checkpoint(txt_filename: str) -> Tuple[Dict[str, Dict], int]:
    """Load checkpoint from disk. Returns (processed_packages, last_index)."""
    checkpoint_file = get_checkpoint_file(txt_filename)
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('packages', {}), data.get('last_index', 0)
        except (json.JSONDecodeError, KeyError) as e:
            safe_print(f"Warning: Failed to load checkpoint: {e}")
    return {}, 0


def clear_checkpoint(txt_filename: str):
    """Remove checkpoint file after successful completion."""
    checkpoint_file = get_checkpoint_file(txt_filename)
    if checkpoint_file.exists():
        checkpoint_file.unlink()


# ============================================================================
# GITHUB METRICS MINER
# ============================================================================

class GitHubMetricsMiner:
    """Mines metrics from GitHub repositories with parallel processing."""

    def __init__(self, token_manager: TokenManager):
        """
        Initialize the miner.

        Args:
            token_manager: TokenManager instance for handling GitHub tokens
        """
        self.token_manager = token_manager
        self.base_url = "https://api.github.com"

    def parse_github_url(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Parse GitHub URL to extract owner and repo name.

        Args:
            url: GitHub repository URL

        Returns:
            Tuple of (owner, repo) or None if parsing fails
        """
        if not url:
            return None

        # Handle various GitHub URL formats
        url = url.rstrip("/")

        # Remove .git suffix if present
        if url.endswith(".git"):
            url = url[:-4]

        # Extract from https://github.com/owner/repo format
        parts = url.split("github.com/")
        if len(parts) < 2:
            return None

        path_parts = parts[1].split("/")
        if len(path_parts) < 2:
            return None

        owner = path_parts[0]
        repo = path_parts[1]

        # Remove any remaining path components or query parameters
        repo = repo.split("/")[0].split("?")[0].split("#")[0]

        return (owner, repo)

    def _make_request(self, url: str, method: str = 'GET', json_data: dict = None, 
                      timeout: int = 10) -> Optional[requests.Response]:
        """Make an API request with retry logic and token rotation."""
        retry_count = 0
        
        while retry_count <= MAX_RETRIES:
            try:
                # Get a token (ensures rate limit is available)
                token = self.token_manager.ensure_rate_limit()
                headers = {'Authorization': f'token {token}'} if token else {}
                
                if method == 'GET':
                    response = requests.get(url, headers=headers, timeout=timeout)
                elif method == 'POST':
                    response = requests.post(url, headers=headers, json=json_data, timeout=timeout)
                else:
                    return None
                
                # Check rate limit and rotate if needed
                if response.status_code == 403:
                    remaining = response.headers.get('X-RateLimit-Remaining', '1')
                    if remaining == '0':
                        if self.token_manager.rotate_token():
                            retry_count += 1
                            continue
                        else:
                            self.token_manager.wait_for_rate_limit_reset()
                            retry_count += 1
                            continue
                
                return response
                
            except (requests.exceptions.Timeout, 
                    requests.exceptions.ConnectionError) as e:
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    wait_time = RETRY_DELAY ** retry_count
                    time.sleep(wait_time)
                else:
                    return None
            except Exception:
                return None
        
        return None

    def get_languages(self, owner: str, repo: str) -> Dict[str, int]:
        """
        Get language breakdown for a repository.
        
        Returns:
            Dictionary mapping language name to bytes of code
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/languages"
        response = self._make_request(url)
        
        if response and response.status_code == 200:
            return response.json()
        return {}

    def calculate_language_proportions(self, languages: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate language proportions as percentages.
        
        Args:
            languages: Dictionary mapping language name to bytes of code
            
        Returns:
            Dictionary mapping language name to percentage (0-100)
        """
        total = sum(languages.values())
        if total == 0:
            return {}
        
        proportions = {}
        for lang, bytes_count in languages.items():
            proportions[lang] = round((bytes_count / total) * 100, 2)
        
        return proportions

    def format_language_string(self, languages: Dict[str, int]) -> str:
        """
        Format languages as a string for CSV output.
        Format: "Language1:XX.XX%;Language2:YY.YY%"
        
        Args:
            languages: Dictionary mapping language name to bytes of code
            
        Returns:
            Formatted string of language proportions
        """
        proportions = self.calculate_language_proportions(languages)
        if not proportions:
            return ""
        
        # Sort by percentage descending
        sorted_langs = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
        return ";".join(f"{lang}:{pct}%" for lang, pct in sorted_langs)

    def get_top_language(self, languages: Dict[str, int]) -> str:
        """Get the top language by bytes of code."""
        if not languages:
            return ""
        return max(languages.keys(), key=lambda k: languages[k])

    def is_forked(self, owner: str, repo: str) -> bool:
        """Check if a repository is forked."""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        response = self._make_request(url)
        
        if response and response.status_code == 200:
            repo_data = response.json()
            return repo_data.get("fork", False)
        return False

    def get_metrics(self, owner: str, repo: str) -> Optional[Dict]:
        """
        Get metrics for a GitHub repository including language proportions.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dictionary with metrics or None if failed
        """
        try:
            # Get repository info (stars, fork status)
            repo_url = f"{self.base_url}/repos/{owner}/{repo}"
            repo_response = self._make_request(repo_url)

            if not repo_response:
                return None
            
            if repo_response.status_code == 404:
                return None
            elif repo_response.status_code == 403:
                return None
            elif repo_response.status_code != 200:
                return None

            repo_data = repo_response.json()

            # Check if forked
            if repo_data.get("fork", False):
                return None

            # Get basic metrics
            stars = repo_data.get("stargazers_count", 0)
            forks = repo_data.get("forks_count", 0)

            # Get language data
            languages = self.get_languages(owner, repo)
            language_string = self.format_language_string(languages)
            top_language = self.get_top_language(languages)

            # Get commits count
            commits = self._get_commits_count(owner, repo)

            # Get pull requests
            prs = self._get_pull_requests(owner, repo)

            # Get issues
            issues = self._get_issues(owner, repo)

            # Get contributors count
            contributors = self._get_contributors_count(owner, repo)

            # Get dependencies and dependents
            dependencies = self._get_dependencies_count(owner, repo)
            dependents = self._get_dependents_count(owner, repo)

            metrics = {
                "stars": stars,
                "forks": forks,
                "commits": commits,
                "active_pull_requests": prs["active"],
                "closed_pull_requests": prs["closed"],
                "all_pull_requests": prs["all"],
                "contributors": contributors,
                "active_issues": issues["active"],
                "closed_issues": issues["closed"],
                "all_issues": issues["all"],
                "dependencies": dependencies,
                "dependents": dependents,
                "top_language": top_language,
                "language_proportions": language_string,
            }

            return metrics

        except Exception:
            return None

    def _get_commits_count(self, owner: str, repo: str) -> int:
        """Get total commits count for a repository using GraphQL API."""
        try:
            graphql_url = "https://api.github.com/graphql"
            query = """
            query($owner: String!, $repo: String!) {
              repository(owner: $owner, name: $repo) {
                defaultBranchRef {
                  target {
                    ... on Commit {
                      history {
                        totalCount
                      }
                    }
                  }
                }
              }
            }
            """
            
            response = self._make_request(
                graphql_url, 
                method='POST',
                json_data={"query": query, "variables": {"owner": owner, "repo": repo}}
            )
            
            if response and response.status_code == 200:
                data = response.json()
                if "data" in data and data["data"]["repository"]:
                    default_branch = data["data"]["repository"]["defaultBranchRef"]
                    if default_branch and default_branch["target"]:
                        return default_branch["target"]["history"]["totalCount"]
            
            return 0

        except Exception:
            return 0

    def _get_pull_requests(self, owner: str, repo: str) -> Dict[str, int]:
        """Get pull request counts for a repository using GraphQL API."""
        try:
            graphql_url = "https://api.github.com/graphql"
            query = """
            query($owner: String!, $repo: String!) {
              repository(owner: $owner, name: $repo) {
                openPullRequests: pullRequests(states: OPEN) {
                  totalCount
                }
                closedPullRequests: pullRequests(states: [CLOSED, MERGED]) {
                  totalCount
                }
              }
            }
            """
            
            response = self._make_request(
                graphql_url,
                method='POST',
                json_data={"query": query, "variables": {"owner": owner, "repo": repo}}
            )
            
            if response and response.status_code == 200:
                data = response.json()
                if "data" in data and data["data"]["repository"]:
                    repo_data = data["data"]["repository"]
                    active = repo_data["openPullRequests"]["totalCount"]
                    closed = repo_data["closedPullRequests"]["totalCount"]
                    return {
                        "active": active,
                        "closed": closed,
                        "all": active + closed,
                    }
            
            return {"active": 0, "closed": 0, "all": 0}

        except Exception:
            return {"active": 0, "closed": 0, "all": 0}

    def _get_issues(self, owner: str, repo: str) -> Dict[str, int]:
        """Get issue counts for a repository using GraphQL API (excludes PRs)."""
        try:
            graphql_url = "https://api.github.com/graphql"
            query = """
            query($owner: String!, $repo: String!) {
              repository(owner: $owner, name: $repo) {
                openIssues: issues(states: OPEN) {
                  totalCount
                }
                closedIssues: issues(states: CLOSED) {
                  totalCount
                }
              }
            }
            """
            
            response = self._make_request(
                graphql_url,
                method='POST',
                json_data={"query": query, "variables": {"owner": owner, "repo": repo}}
            )
            
            if response and response.status_code == 200:
                data = response.json()
                if "data" in data and data["data"]["repository"]:
                    repo_data = data["data"]["repository"]
                    active = repo_data["openIssues"]["totalCount"]
                    closed = repo_data["closedIssues"]["totalCount"]
                    return {
                        "active": active,
                        "closed": closed,
                        "all": active + closed,
                    }
            
            return {"active": 0, "closed": 0, "all": 0}

        except Exception:
            return {"active": 0, "closed": 0, "all": 0}

    def _get_contributors_count(self, owner: str, repo: str) -> int:
        """Get contributors count for a repository."""
        try:
            contributors_url = f"{self.base_url}/repos/{owner}/{repo}/contributors"
            params = {"per_page": 1, "anon": "true"}
            
            token = self.token_manager.ensure_rate_limit()
            headers = {'Authorization': f'token {token}'} if token else {}
            
            response = requests.get(contributors_url, headers=headers, params=params, timeout=10)

            if response.status_code == 200 and "Link" in response.headers:
                link_header = response.headers["Link"]
                match = re.search(r'page=(\d+)>; rel="last"', link_header)
                if match:
                    return int(match.group(1))
            elif response.status_code == 200:
                return len(response.json())

            return 0

        except Exception:
            return 0

    def _get_dependencies_count(self, owner: str, repo: str) -> int:
        """Get dependencies count by scraping the GitHub dependency graph page."""
        try:
            dependencies_url = f"https://github.com/{owner}/{repo}/network/dependencies"
            
            token = self.token_manager.ensure_rate_limit()
            headers = {'Authorization': f'token {token}'} if token else {}
            
            response = requests.get(dependencies_url, headers=headers, timeout=10)

            if response.status_code == 200:
                content = response.text
                match = re.search(r'([\d,]+)\s+Total', content)
                if match:
                    count_str = match.group(1).replace(',', '')
                    return int(count_str)
                match = re.search(r'(\d+)\s+Dependenc(?:y|ies)', content, re.IGNORECASE)
                if match:
                    return int(match.group(1))

            return 0

        except Exception:
            return 0

    def _get_dependents_count(self, owner: str, repo: str) -> int:
        """Get dependents count by scraping the GitHub network/dependents page."""
        try:
            dependents_url = f"https://github.com/{owner}/{repo}/network/dependents"
            
            token = self.token_manager.ensure_rate_limit()
            headers = {'Authorization': f'token {token}'} if token else {}
            
            response = requests.get(dependents_url, headers=headers, timeout=10)

            if response.status_code == 200:
                content = response.text
                match = re.search(r'([\d,]+)\s+Repositor(?:y|ies)', content)
                if match:
                    count_str = match.group(1).replace(',', '')
                    return int(count_str)
                match = re.search(r'"dependents_count":(\d+)', content)
                if match:
                    return int(match.group(1))

            return 0

        except Exception:
            return 0


# ============================================================================
# FILE PARSING
# ============================================================================

def parse_txt_file(txt_path: str) -> List[Dict[str, str]]:
    """
    Parse a txt file from Directory-Structure-Miner results.

    Args:
        txt_path: Path to the txt file

    Returns:
        List of dictionaries with owner/repo and repository URL
    """
    packages = []

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by package sections
    package_sections = re.split(r"={80}\nPackage \d+/\d+\n={80}\n", content)

    for section in package_sections[1:]:  # Skip header
        if not section.strip():
            continue

        # Extract repository URL
        repo_match = re.search(r"Repository: (.+)", section)
        if not repo_match:
            continue

        repo_url = repo_match.group(1).strip()

        # Extract Owner/Repo
        owner_repo_match = re.search(r"Owner/Repo: (.+)", section)
        if not owner_repo_match:
            continue

        owner_repo = owner_repo_match.group(1).strip()

        # Extract Ecosystems
        ecosystems_match = re.search(r"Ecosystems: (.+)", section)
        ecosystems = ecosystems_match.group(1).strip() if ecosystems_match else ""

        packages.append({
            "owner_repo": owner_repo,
            "repo_url": repo_url,
            "ecosystems": ecosystems
        })

    return packages


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def process_single_package(package: Dict, miner: GitHubMetricsMiner) -> Optional[Dict]:
    """
    Process a single package to get metrics.
    
    Args:
        package: Dictionary with 'owner_repo', 'repo_url', and 'ecosystems'
        miner: GitHubMetricsMiner instance
        
    Returns:
        Dictionary with metrics or None if failed
    """
    owner_repo = package["owner_repo"]
    repo_url = package["repo_url"]
    ecosystems = package.get("ecosystems", "")
    
    # Parse GitHub URL
    parsed = miner.parse_github_url(repo_url)
    if not parsed:
        return None
    
    owner, repo_name = parsed
    
    # Get metrics
    metrics = miner.get_metrics(owner, repo_name)
    
    if metrics:
        return {
            "owner_repo": owner_repo,
            "repo_url": repo_url,
            "ecosystems": ecosystems,
            **metrics
        }
    
    return None


def process_packages_parallel(
    all_packages: List[Dict],
    miner: GitHubMetricsMiner,
    processed_packages: Dict,
    max_workers: int = MAX_WORKERS,
) -> Dict[str, Dict]:
    """
    Process all packages using parallel workers.

    Args:
        all_packages: List of all packages to process
        miner: GitHubMetricsMiner instance
        processed_packages: Dictionary of already processed packages
        max_workers: Number of parallel workers
        
    Returns:
        Dictionary mapping package keys to results
    """
    # Filter out already processed packages
    packages_to_process = []
    for i, pkg in enumerate(all_packages):
        pkg_key = f"{pkg['owner_repo']}|{pkg['repo_url']}"
        if pkg_key not in processed_packages:
            packages_to_process.append((i, pkg))

    if not packages_to_process:
        safe_print(f"All packages already processed")
        return processed_packages

    safe_print(f"Packages to process: {len(packages_to_process)}")
    safe_print(f"Using {max_workers} parallel workers\n")

    # Process packages in parallel with progress bar
    results_lock = Lock()
    checkpoint_interval = 100  # Save checkpoint every N packages
    processed_count = 0

    with tqdm(total=len(packages_to_process), desc=f"Mining repositories", unit="repo") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pkg = {
                executor.submit(process_single_package, pkg, miner): (idx, pkg)
                for idx, pkg in packages_to_process
            }

            for future in as_completed(future_to_pkg):
                idx, pkg = future_to_pkg[future]
                pkg_key = f"{pkg['owner_repo']}|{pkg['repo_url']}"
                
                try:
                    result = future.result()
                    
                    with results_lock:
                        if result:
                            processed_packages[pkg_key] = result
                        else:
                            # Mark as processed but failed
                            processed_packages[pkg_key] = None
                        
                        processed_count += 1
                        
                        # Save checkpoint periodically
                        if processed_count % checkpoint_interval == 0:
                            save_checkpoint("global", processed_packages, idx)
                
                except Exception as e:
                    with results_lock:
                        processed_packages[pkg_key] = None
                
                pbar.update(1)

    # Save final checkpoint
    save_checkpoint("global", processed_packages, len(all_packages) - 1)
    
    return processed_packages


def write_final_csv(output_path: str, all_packages: List[Dict], processed_packages: Dict) -> int:
    """
    Write final results to a single CSV file.
    
    Args:
        output_path: Path to output CSV file
        all_packages: List of all packages (for ordering and ensuring all are included)
        processed_packages: Dictionary of processed results
        
    Returns:
        Number of successfully mined packages
    """
    fieldnames = [
        "owner_repo",
        "repo_url",
        "ecosystems",
        "stars",
        "forks",
        "commits",
        "active_pull_requests",
        "closed_pull_requests",
        "all_pull_requests",
        "contributors",
        "active_issues",
        "closed_issues",
        "all_issues",
        "dependencies",
        "dependents",
        "top_language",
        "language_proportions",
    ]
    
    successful_count = 0
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pkg in all_packages:
            pkg_key = f"{pkg['owner_repo']}|{pkg['repo_url']}"
            result = processed_packages.get(pkg_key)
            
            if result:
                writer.writerow(result)
                successful_count += 1
    
    return successful_count


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mine GitHub metrics from Directory-Structure-Miner results (with parallel processing)"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing txt input files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output file (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output CSV filename (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--token",
        nargs="+",
        help="GitHub personal access token(s) (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific txt files to process (default: all txt files in input dir)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})",
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"GitHub Metrics Miner (Parallel Processing)")
    print(f"{'=' * 80}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {args.workers}")

    # Collect tokens
    tokens = args.token or []
    
    # Check for environment variable
    env_token = os.environ.get('GITHUB_TOKEN', '').strip()
    if env_token and not tokens:
        if "," in env_token:
            tokens = [t.strip() for t in env_token.split(",") if t.strip()]
        else:
            tokens = [env_token]
        print(f"\nUsing {len(tokens)} token(s) from GITHUB_TOKEN environment variable")
    
    # Prompt for tokens if not provided
    if not tokens:
        print("\n" + "=" * 80)
        print("GitHub Token Input")
        print("=" * 80)
        print("Enter your GitHub token(s). You can:")
        print("  • Enter multiple tokens separated by commas")
        print("  • Press Enter to use unauthenticated API (60 requests/hour)")
        print("  • Press Ctrl+C to cancel")
        print("=" * 80)

        try:
            token_input = input("\nEnter token(s): ").strip()
            if token_input:
                tokens = [t.strip() for t in re.split(r'[\s,]+', token_input) if t.strip()]
        except (KeyboardInterrupt, EOFError):
            print("\n\n✗ Operation cancelled by user.")
            return

    # Initialize token manager
    token_manager = TokenManager(tokens)

    if token_manager.tokens:
        token_manager.validate_tokens()
        try:
            response = (
                input("Do you want to proceed with these tokens? (yes/no): ")
                .strip()
                .lower()
            )
            if response not in ["yes", "y"]:
                print("\n✗ Operation cancelled by user.")
                return
        except (KeyboardInterrupt, EOFError):
            print("\n\n✗ Operation cancelled by user.")
            return
    else:
        print("\n" + "!" * 80)
        print("⚠  WARNING: No GitHub Token Provided")
        print("!" * 80)
        print("\nRunning without authentication has severe limitations:")
        print("  • Rate limit: Only 60 requests per hour (vs 5,000 with token)")
        print("  • Cannot access private repositories")
        print("  • May encounter frequent rate limit errors")
        print("  • Processing will be significantly slower")
        print("\nTo use a token:")
        print("  • Set GITHUB_TOKEN environment variable, or")
        print("  • Use --token argument with your personal access token")
        print("\nGet a token at: https://github.com/settings/tokens")
        print("=" * 80)

        try:
            response = (
                input("\nDo you want to continue without a token? (yes/no): ")
                .strip()
                .lower()
            )
            if response not in ["yes", "y"]:
                print("\n✗ Operation cancelled by user.")
                return
            print("\n✓ Continuing without token...")
        except (KeyboardInterrupt, EOFError):
            print("\n\n✗ Operation cancelled by user.")
            return

    # Initialize miner
    miner = GitHubMetricsMiner(token_manager)

    # Get input directory
    input_dir = (script_dir / args.input_dir).resolve()
    print(f"Input directory: {input_dir}")
    
    # Output file path
    output_path = output_dir / args.output_file
    print(f"Output file: {output_path}")

    # Get txt files to process
    txt_files = []
    if args.files:
        # If specific files provided
        for f in args.files:
            file_path = input_dir / f
            if file_path.is_file():
                txt_files.append(file_path)
            else:
                # Try searching in subdirectories
                for subdir in input_dir.iterdir():
                    if subdir.is_dir():
                        possible_path = subdir / f
                        if possible_path.is_file():
                            txt_files.append(possible_path)
                            break
    else:
        # Get all txt files from subdirectories
        ecosystem_dirs = [
            d
            for d in input_dir.iterdir()
            if d.is_dir() and d.name.endswith("_ecosystems")
        ]

        if ecosystem_dirs:
            # Structure with ecosystem folders
            for ecosystem_dir in sorted(ecosystem_dirs):
                txt_files.extend(list(ecosystem_dir.glob("*.txt")))
        else:
            # Fallback: txt files directly in input_dir
            txt_files = list(input_dir.glob("*.txt"))

    txt_files = [f for f in txt_files if f.is_file()]

    if not txt_files:
        print(f"\n✗ No txt files found in {input_dir}")
        return

    print(f"\nFound {len(txt_files)} txt file(s) to process:")

    # Group by ecosystem count for display
    files_by_count = {}
    for f in txt_files:
        # Determine ecosystem count from parent folder name
        parent = f.parent.name
        if parent.endswith("_ecosystems"):
            count = int(parent.split("_")[0])
        else:
            # Estimate from filename
            count = len(f.stem.split("_"))
        
        if count not in files_by_count:
            files_by_count[count] = []
        files_by_count[count].append(f)

    for count in sorted(files_by_count.keys()):
        print(
            f"\n  {count}-ecosystem combinations ({len(files_by_count[count])} files):"
        )
        for f in sorted(files_by_count[count])[:5]:  # Show first 5
            print(f"    • {f.name}")
        if len(files_by_count[count]) > 5:
            print(f"    ... and {len(files_by_count[count]) - 5} more")

    # Load checkpoint if exists
    processed_packages, _ = load_checkpoint("global")
    if processed_packages:
        safe_print(f"\nResuming from checkpoint: {len(processed_packages)} packages already processed")

    # Collect all packages from all files
    safe_print(f"\n{'=' * 80}")
    safe_print("Parsing input files...")
    safe_print(f"{'=' * 80}")
    
    all_packages = []
    files_by_count_stats = {}
    
    for txt_file in tqdm(txt_files, desc="Parsing files", unit="file"):
        try:
            packages = parse_txt_file(str(txt_file))
            all_packages.extend(packages)
            
            # Track stats by ecosystem count
            parent = txt_file.parent.name
            if parent.endswith("_ecosystems"):
                count = int(parent.split("_")[0])
            else:
                count = len(txt_file.stem.split("_"))
            
            if count not in files_by_count_stats:
                files_by_count_stats[count] = {"files": 0, "packages": 0}
            files_by_count_stats[count]["files"] += 1
            files_by_count_stats[count]["packages"] += len(packages)
            
        except Exception as e:
            safe_print(f"\n✗ Error parsing {txt_file.name}: {e}")
            continue

    # Remove duplicates (same owner/repo)
    unique_packages = {}
    for pkg in all_packages:
        key = pkg["owner_repo"]
        if key not in unique_packages:
            unique_packages[key] = pkg
    
    all_packages = list(unique_packages.values())
    
    safe_print(f"\nTotal unique repositories: {len(all_packages)}")
    safe_print(f"Files parsed: {len(txt_files)}")
    
    # Display stats by ecosystem count
    safe_print(f"\nPackages by ecosystem count:")
    for count in sorted(files_by_count_stats.keys()):
        stats = files_by_count_stats[count]
        safe_print(f"  {count}-ecosystems: {stats['packages']} packages from {stats['files']} files")

    # Process all packages
    try:
        safe_print(f"\n{'=' * 80}")
        safe_print("Mining GitHub metrics...")
        safe_print(f"{'=' * 80}\n")
        
        processed_packages = process_packages_parallel(
            all_packages,
            miner,
            processed_packages,
            max_workers=args.workers,
        )
        
        # Write final CSV
        safe_print(f"\n{'=' * 80}")
        safe_print("Writing results to CSV...")
        safe_print(f"{'=' * 80}")
        
        successful_count = write_final_csv(str(output_path), all_packages, processed_packages)
        
        # Clear checkpoint after successful completion
        clear_checkpoint("global")
        
        safe_print(f"\n✓ Results saved to {output_path}")
        safe_print(f"  Successfully mined: {successful_count}/{len(all_packages)} repositories")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Progress has been saved.")
        print("Run the script again to resume from checkpoint.")
        return
    except Exception as e:
        safe_print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    print(f"\n{'=' * 80}")
    print("Processing Complete!")
    print(f"{'=' * 80}")
    print(f"Output file: {output_path}")
    print(f"Total repositories processed: {len(all_packages)}")
    print(f"Successfully mined: {successful_count}")
    print(f"Success rate: {(successful_count/len(all_packages)*100):.1f}%")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
