#!/usr/bin/env python3
"""
GitHub Metrics Miner
Mines GitHub repository metrics from Common-Package-Filter CSV results.
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
DEFAULT_INPUT_DIR = DATASET_DIR / "Multirepo-Common-Package-Filter"
DEFAULT_OUTPUT_DIR = DATASET_DIR / "Metric-Miner-Github"
DEFAULT_OUTPUT_FILE = "github_metrics.json"
DEFAULT_CACHE_DIR = DATASET_DIR / "Cache/Metric-Miner-Github"
DEFAULT_CACHE_FILE = "github_metrics.json"
LOG_FILE = SCRIPT_DIR / "processing.log"

# ============================================================================
# PARALLEL PROCESSING CONFIGURATION
# ============================================================================

MAX_WORKERS = 2  # Number of concurrent threads for API calls (reduced to avoid rate limits)
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
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_progress(output_path: Path, processed_packages: Dict[str, Dict], last_index: int):
    """Save progress directly to output JSON file."""
    progress_data = {
        'last_index': last_index,
        'processed_count': len([v for v in processed_packages.values() if v is not None]),
        'timestamp': time.time(),
        'packages': processed_packages
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2)


def load_progress(output_path: Path) -> Tuple[Dict[str, Dict], int]:
    """Load progress from output JSON file. Returns (processed_packages, last_index)."""
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('packages', {}), data.get('last_index', 0)
        except (json.JSONDecodeError, KeyError) as e:
            safe_print(f"Warning: Failed to load progress: {e}")
    return {}, 0


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def load_cache(cache_path: Path) -> Dict[str, Dict]:
    """
    Load cache from a previous metrics mining run.
    
    Args:
        cache_path: Path to the cache JSON file
        
    Returns:
        Dictionary mapping package keys (owner_repo|repo_url) to cached metrics
    """
    if not cache_path.exists():
        safe_print(f"Cache file not found: {cache_path}")
        return {}
    
    try:
        safe_print(f"Loading cache from: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            packages = data.get('packages', {})
            safe_print(f"  Loaded {len(packages)} cached entries")
            return packages
    except (json.JSONDecodeError, KeyError) as e:
        safe_print(f"Warning: Failed to load cache: {e}")
        return {}


def build_cache_index(cache: Dict[str, Dict]) -> Dict[str, str]:
    """
    Build a search index from cache for fast lookup.
    Creates multiple lookup keys for each cached entry.
    
    Args:
        cache: Dictionary of cached packages (key -> metrics)
        
    Returns:
        Dictionary mapping various lookup keys to the original cache key
    """
    index = {}
    
    for cache_key, metrics in cache.items():
        if metrics is None:
            continue
            
        # Primary key: owner_repo|repo_url (already the format used)
        index[cache_key] = cache_key
        
        # Extract owner_repo and repo_url from cache key
        parts = cache_key.split('|', 1)
        if len(parts) == 2:
            owner_repo, repo_url = parts
            
            # Index by owner_repo only
            if owner_repo not in index:
                index[owner_repo] = cache_key
            
            # Index by repo_url only
            if repo_url not in index:
                index[repo_url] = cache_key
            
            # Index by normalized repo_url (lowercase, no trailing slash)
            normalized_url = repo_url.lower().rstrip('/')
            if normalized_url not in index:
                index[normalized_url] = cache_key
                
            # Index by owner_repo from metrics if present
            if 'owner_repo' in metrics and metrics['owner_repo']:
                metrics_owner_repo = metrics['owner_repo']
                if metrics_owner_repo not in index:
                    index[metrics_owner_repo] = cache_key
                # Also lowercase version
                if metrics_owner_repo.lower() not in index:
                    index[metrics_owner_repo.lower()] = cache_key
    
    return index


def lookup_in_cache(pkg: Dict, cache: Dict[str, Dict], cache_index: Dict[str, str]) -> Optional[Dict]:
    """
    Look up a package in the cache using multiple strategies.
    
    Args:
        pkg: Package dictionary with 'owner_repo' and 'repo_url'
        cache: The full cache dictionary
        cache_index: The search index built from cache
        
    Returns:
        Cached metrics if found, None otherwise
    """
    owner_repo = pkg.get('owner_repo', '')
    repo_url = pkg.get('repo_url', '')
    
    # Try primary key format first
    primary_key = f"{owner_repo}|{repo_url}"
    if primary_key in cache_index:
        cache_key = cache_index[primary_key]
        return cache.get(cache_key)
    
    # Try owner_repo only
    if owner_repo and owner_repo in cache_index:
        cache_key = cache_index[owner_repo]
        return cache.get(cache_key)
    
    # Try owner_repo lowercase
    if owner_repo and owner_repo.lower() in cache_index:
        cache_key = cache_index[owner_repo.lower()]
        return cache.get(cache_key)
    
    # Try repo_url
    if repo_url and repo_url in cache_index:
        cache_key = cache_index[repo_url]
        return cache.get(cache_key)
    
    # Try normalized repo_url
    if repo_url:
        normalized_url = repo_url.lower().rstrip('/')
        if normalized_url in cache_index:
            cache_key = cache_index[normalized_url]
            return cache.get(cache_key)
    
    return None


def transfer_from_cache(
    all_packages: List[Dict],
    cache: Dict[str, Dict],
    cache_index: Dict[str, str],
    processed_packages: Dict[str, Dict]
) -> Tuple[Dict[str, Dict], List[Dict], int]:
    """
    Transfer cached packages to processed_packages and identify remaining packages.
    
    Args:
        all_packages: List of all packages to process
        cache: The full cache dictionary
        cache_index: The search index built from cache
        processed_packages: Dictionary to store results
        
    Returns:
        Tuple of (updated processed_packages, remaining packages list, cache hit count)
    """
    remaining_packages = []
    cache_hits = 0
    
    for pkg in all_packages:
        pkg_key = f"{pkg['owner_repo']}|{pkg['repo_url']}"
        
        # Skip if already processed
        if pkg_key in processed_packages:
            continue
        
        # Try to find in cache
        cached_result = lookup_in_cache(pkg, cache, cache_index)
        
        if cached_result:
            # Update ecosystems from current package (may differ from cache)
            result = cached_result.copy()
            result['ecosystems'] = pkg.get('ecosystems', cached_result.get('ecosystems', ''))
            result['from_cache'] = True
            processed_packages[pkg_key] = result
            cache_hits += 1
        else:
            remaining_packages.append(pkg)
    
    return processed_packages, remaining_packages, cache_hits


# ============================================================================
# GITHUB METRICS MINER
# ============================================================================

class GitHubMetricsMiner:
    """Mines metrics from GitHub repositories with parallel processing."""

    def __init__(self, token_manager: TokenManager, logger=None):
        """
        Initialize the miner.

        Args:
            token_manager: TokenManager instance for handling GitHub tokens
            logger: Logger instance for logging
        """
        self.token_manager = token_manager
        self.base_url = "https://api.github.com"
        self.logger = logger

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
                
                # Handle rate limiting
                if response.status_code == 429:
                    # Explicit rate limit exceeded
                    if self.logger:
                        log_message(self.logger, 'warning',
                                   f"Rate limit exceeded (429). Attempting token rotation...")
                    if self.token_manager.rotate_token():
                        time.sleep(1)  # Brief pause after rotation
                        retry_count += 1
                        continue
                    else:
                        # No other tokens available, wait for reset
                        self.token_manager.wait_for_rate_limit_reset()
                        retry_count += 1
                        continue
                
                # Check rate limit and rotate if needed
                if response.status_code == 403:
                    # For GraphQL, 403 without rate limit headers means auth required
                    if method == 'POST' and 'graphql' in url:
                        # GraphQL requires authentication
                        try:
                            error_data = response.json()
                            if 'errors' in error_data:
                                # Authentication error, don't retry
                                return response
                        except:
                            pass
                    
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
                return {"error": "API_REQUEST_FAILED", "error_detail": "No response from API"}
            
            if repo_response.status_code == 404:
                # HTTP 404 is not critical - repository may be deleted or made private
                return {"error": "HTTP_404", "error_detail": "Repository not found (deleted or private)"}
            elif repo_response.status_code == 403:
                return {"error": "HTTP_403", "error_detail": "Access forbidden (private or rate limit)"}
            elif repo_response.status_code != 200:
                return {"error": f"HTTP_{repo_response.status_code}", "error_detail": f"API returned status {repo_response.status_code}"}

            repo_data = repo_response.json()

            # Get basic metrics (including fork and archived status)
            stars = repo_data.get("stargazers_count", 0)
            forks = repo_data.get("forks_count", 0)
            is_fork = repo_data.get("fork", False)
            is_archived = repo_data.get("archived", False)

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
                "is_fork": is_fork,
                "is_archived": is_archived,
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

        except Exception as e:
            if self.logger:
                log_message(self.logger, 'error',
                           f"Exception in get_metrics for {owner}/{repo}: {type(e).__name__}: {str(e)}")
            return None

    def _get_commits_count(self, owner: str, repo: str) -> int:
        """Get total commits count for a repository using GraphQL API with retry logic."""
        for attempt in range(MAX_RETRIES + 1):
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
                    
                    # Check for GraphQL errors (e.g., authentication failures)
                    if "errors" in data:
                        if self.logger:
                            errors = data.get("errors", [])
                            error_msg = errors[0].get("message", "Unknown error") if errors else "Unknown error"
                            log_message(self.logger, 'error',
                                       f"GraphQL error for {owner}/{repo} commits: {error_msg}")
                        return 0
                    
                    # Verify data structure
                    if "data" in data and data["data"] and data["data"]["repository"]:
                        default_branch = data["data"]["repository"]["defaultBranchRef"]
                        if default_branch and default_branch["target"] and "history" in default_branch["target"]:
                            count = default_branch["target"]["history"]["totalCount"]
                            # Verify count is valid
                            if isinstance(count, int) and count >= 0:
                                return count
                            else:
                                if self.logger:
                                    log_message(self.logger, 'warning',
                                               f"Invalid commits count for {owner}/{repo}: {count}")
                
                # If response not successful and not last attempt, retry
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY ** (attempt + 1)
                    if self.logger:
                        log_message(self.logger, 'info',
                                   f"Retrying commits count for {owner}/{repo} (attempt {attempt + 2}/{MAX_RETRIES + 1})")
                    time.sleep(wait_time)
                    continue
                
                return 0

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY ** (attempt + 1)
                    if self.logger:
                        log_message(self.logger, 'warning',
                                   f"Error getting commits for {owner}/{repo} (attempt {attempt + 1}): {str(e)}. Retrying...")
                    time.sleep(wait_time)
                else:
                    if self.logger:
                        log_message(self.logger, 'error',
                                   f"Failed to get commits for {owner}/{repo} after {MAX_RETRIES + 1} attempts: {str(e)}")
                    return 0
        
        return 0
    def _get_pull_requests(self, owner: str, repo: str) -> Dict[str, int]:
        """Get pull request counts for a repository using GraphQL API with retry logic."""
        for attempt in range(MAX_RETRIES + 1):
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
                    
                    # Check for GraphQL errors
                    if "errors" in data:
                        if self.logger:
                            errors = data.get("errors", [])
                            error_msg = errors[0].get("message", "Unknown error") if errors else "Unknown error"
                            log_message(self.logger, 'error',
                                       f"GraphQL error for {owner}/{repo} PRs: {error_msg}")
                        return {"active": 0, "closed": 0, "all": 0}
                    
                    # Verify data structure
                    if "data" in data and data["data"] and data["data"]["repository"]:
                        repo_data = data["data"]["repository"]
                        if "openPullRequests" in repo_data and "closedPullRequests" in repo_data:
                            active = repo_data["openPullRequests"]["totalCount"]
                            closed = repo_data["closedPullRequests"]["totalCount"]
                            # Verify counts are valid
                            if isinstance(active, int) and isinstance(closed, int) and active >= 0 and closed >= 0:
                                return {
                                    "active": active,
                                    "closed": closed,
                                    "all": active + closed,
                                }
                
                # If response not successful and not last attempt, retry
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY ** (attempt + 1)
                    if self.logger:
                        log_message(self.logger, 'info',
                                   f"Retrying PRs for {owner}/{repo} (attempt {attempt + 2}/{MAX_RETRIES + 1})")
                    time.sleep(wait_time)
                    continue
                
                return {"active": 0, "closed": 0, "all": 0}

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY ** (attempt + 1)
                    if self.logger:
                        log_message(self.logger, 'warning',
                                   f"Error getting PRs for {owner}/{repo} (attempt {attempt + 1}): {str(e)}. Retrying...")
                    time.sleep(wait_time)
                else:
                    if self.logger:
                        log_message(self.logger, 'error',
                                   f"Failed to get PRs for {owner}/{repo} after {MAX_RETRIES + 1} attempts: {str(e)}")
                    return {"active": 0, "closed": 0, "all": 0}
        
        return {"active": 0, "closed": 0, "all": 0}

    def _get_issues(self, owner: str, repo: str) -> Dict[str, int]:
        """Get issue counts for a repository using GraphQL API (excludes PRs) with retry logic."""
        for attempt in range(MAX_RETRIES + 1):
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
                    
                    # Check for GraphQL errors
                    if "errors" in data:
                        if self.logger:
                            errors = data.get("errors", [])
                            error_msg = errors[0].get("message", "Unknown error") if errors else "Unknown error"
                            log_message(self.logger, 'error',
                                       f"GraphQL error for {owner}/{repo} issues: {error_msg}")
                        return {"active": 0, "closed": 0, "all": 0}
                    
                    # Verify data structure
                    if "data" in data and data["data"] and data["data"]["repository"]:
                        repo_data = data["data"]["repository"]
                        if "openIssues" in repo_data and "closedIssues" in repo_data:
                            active = repo_data["openIssues"]["totalCount"]
                            closed = repo_data["closedIssues"]["totalCount"]
                            # Verify counts are valid
                            if isinstance(active, int) and isinstance(closed, int) and active >= 0 and closed >= 0:
                                return {
                                    "active": active,
                                    "closed": closed,
                                    "all": active + closed,
                                }
                
                # If response not successful and not last attempt, retry
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY ** (attempt + 1)
                    if self.logger:
                        log_message(self.logger, 'info',
                                   f"Retrying issues for {owner}/{repo} (attempt {attempt + 2}/{MAX_RETRIES + 1})")
                    time.sleep(wait_time)
                    continue
                
                return {"active": 0, "closed": 0, "all": 0}

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY ** (attempt + 1)
                    if self.logger:
                        log_message(self.logger, 'warning',
                                   f"Error getting issues for {owner}/{repo} (attempt {attempt + 1}): {str(e)}. Retrying...")
                    time.sleep(wait_time)
                else:
                    if self.logger:
                        log_message(self.logger, 'error',
                                   f"Failed to get issues for {owner}/{repo} after {MAX_RETRIES + 1} attempts: {str(e)}")
                    return {"active": 0, "closed": 0, "all": 0}
        
        return {"active": 0, "closed": 0, "all": 0}

    def _get_contributors_count(self, owner: str, repo: str) -> int:
        """Get contributors count for a repository (including anonymous) with retry logic."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                contributors_url = f"{self.base_url}/repos/{owner}/{repo}/contributors"
                params = {"per_page": 1, "anon": "true"}  # Remove anon=true to match GitHub webpage data
                
                token = self.token_manager.ensure_rate_limit()
                headers = {'Authorization': f'token {token}'} if token else {}
                
                response = requests.get(contributors_url, headers=headers, params=params, timeout=10)

                if response.status_code == 200:
                    # Try pagination header first
                    if "Link" in response.headers:
                        link_header = response.headers["Link"]
                        match = re.search(r'page=(\d+)>; rel="last"', link_header)
                        if match:
                            count = int(match.group(1))
                            # Verify count is valid
                            if count > 0:
                                return count
                    # Fallback to counting array length
                    contributors = response.json()
                    if isinstance(contributors, list):
                        return len(contributors)
                    else:
                        if self.logger:
                            log_message(self.logger, 'warning',
                                       f"Invalid contributors response for {owner}/{repo}: not a list")
                elif response.status_code == 404:
                    # Repo not found or no contributors - return 0 without retry
                    return 0
                
                # If response not successful and not last attempt, retry
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY ** (attempt + 1)
                    if self.logger:
                        log_message(self.logger, 'info',
                                   f"Retrying contributors for {owner}/{repo} (attempt {attempt + 2}/{MAX_RETRIES + 1})")
                    time.sleep(wait_time)
                    continue
                
                return 0

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY ** (attempt + 1)
                    if self.logger:
                        log_message(self.logger, 'warning',
                                   f"Error getting contributors for {owner}/{repo} (attempt {attempt + 1}): {str(e)}. Retrying...")
                    time.sleep(wait_time)
                else:
                    if self.logger:
                        log_message(self.logger, 'error',
                                   f"Failed to get contributors for {owner}/{repo} after {MAX_RETRIES + 1} attempts: {str(e)}")
                    return 0
        
        return 0

    def _get_dependencies_count(self, owner: str, repo: str) -> int:
        """Get dependencies count by scraping the GitHub dependency graph page with retry logic."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                dependencies_url = f"https://github.com/{owner}/{repo}/network/dependencies"
                
                token = self.token_manager.ensure_rate_limit()
                headers = {'Authorization': f'token {token}'} if token else {}
                
                response = requests.get(dependencies_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    content = response.text
                    # Try to find "X Total" pattern
                    match = re.search(r'([\d,]+)\s+Total', content)
                    if match:
                        count_str = match.group(1).replace(',', '')
                        count = int(count_str)
                        # Verify count is valid
                        if count >= 0:
                            return count
                        else:
                            if self.logger:
                                log_message(self.logger, 'warning',
                                           f"Invalid dependencies count for {owner}/{repo}: {count}")
                    # Try alternative pattern
                    match = re.search(r'(\d+)\s+Dependenc(?:y|ies)', content, re.IGNORECASE)
                    if match:
                        count = int(match.group(1))
                        if count >= 0:
                            return count
                elif response.status_code == 404:
                    # Repo not found or no dependencies page - return 0 without retry
                    return 0
                elif response.status_code == 429:
                    # Rate limit exceeded - handle specially
                    if self.logger:
                        log_message(self.logger, 'warning',
                                   f"Rate limit hit for {owner}/{repo} dependencies. Waiting for reset...")
                    # Try rotating to another token first
                    if self.token_manager.rotate_token():
                        if attempt < MAX_RETRIES:
                            time.sleep(1)  # Brief pause after token rotation
                            continue
                    else:
                        # No other tokens available, wait for rate limit reset
                        self.token_manager.wait_for_rate_limit_reset()
                        if attempt < MAX_RETRIES:
                            continue
                    return 0
                elif response.status_code != 200:
                    # Log non-200/404/429 errors
                    if attempt < MAX_RETRIES:
                        wait_time = RETRY_DELAY ** (attempt + 1)
                        if self.logger:
                            log_message(self.logger, 'info',
                                       f"Failed to fetch dependencies for {owner}/{repo}: HTTP {response.status_code}. Retrying...")
                        time.sleep(wait_time)
                        continue
                    else:
                        if self.logger:
                            log_message(self.logger, 'warning',
                                       f"Failed to fetch dependencies for {owner}/{repo}: HTTP {response.status_code}")

                return 0

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY ** (attempt + 1)
                    if self.logger:
                        log_message(self.logger, 'warning',
                                   f"Error getting dependencies for {owner}/{repo} (attempt {attempt + 1}): {str(e)}. Retrying...")
                    time.sleep(wait_time)
                else:
                    if self.logger:
                        log_message(self.logger, 'error',
                                   f"Failed to get dependencies for {owner}/{repo} after {MAX_RETRIES + 1} attempts: {str(e)}")
                    return 0
        
        return 0

    def _get_dependents_count(self, owner: str, repo: str) -> int:
        """Get dependents count by scraping the GitHub network/dependents page with retry logic."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                dependents_url = f"https://github.com/{owner}/{repo}/network/dependents"
                
                token = self.token_manager.ensure_rate_limit()
                headers = {'Authorization': f'token {token}'} if token else {}
                
                response = requests.get(dependents_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    content = response.text
                    # Try to find "X Repositories" pattern
                    match = re.search(r'([\d,]+)\s+Repositor(?:y|ies)', content)
                    if match:
                        count_str = match.group(1).replace(',', '')
                        count = int(count_str)
                        # Verify count is valid
                        if count >= 0:
                            return count
                        else:
                            if self.logger:
                                log_message(self.logger, 'warning',
                                           f"Invalid dependents count for {owner}/{repo}: {count}")
                    # Try alternative pattern
                    match = re.search(r'"dependents_count":(\d+)', content)
                    if match:
                        count = int(match.group(1))
                        if count >= 0:
                            return count
                elif response.status_code == 404:
                    # Repo not found or no dependents page - return 0 without retry
                    return 0
                elif response.status_code == 429:
                    # Rate limit exceeded - handle specially
                    if self.logger:
                        log_message(self.logger, 'warning',
                                   f"Rate limit hit for {owner}/{repo} dependents. Waiting for reset...")
                    # Try rotating to another token first
                    if self.token_manager.rotate_token():
                        if attempt < MAX_RETRIES:
                            time.sleep(1)  # Brief pause after token rotation
                            continue
                    else:
                        # No other tokens available, wait for rate limit reset
                        self.token_manager.wait_for_rate_limit_reset()
                        if attempt < MAX_RETRIES:
                            continue
                    return 0
                elif response.status_code != 200:
                    # Log non-200/404/429 errors
                    if attempt < MAX_RETRIES:
                        wait_time = RETRY_DELAY ** (attempt + 1)
                        if self.logger:
                            log_message(self.logger, 'info',
                                       f"Failed to fetch dependents for {owner}/{repo}: HTTP {response.status_code}. Retrying...")
                        time.sleep(wait_time)
                        continue
                    else:
                        if self.logger:
                            log_message(self.logger, 'warning',
                                       f"Failed to fetch dependents for {owner}/{repo}: HTTP {response.status_code}")

                return 0

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY ** (attempt + 1)
                    if self.logger:
                        log_message(self.logger, 'warning',
                                   f"Error getting dependents for {owner}/{repo} (attempt {attempt + 1}): {str(e)}. Retrying...")
                    time.sleep(wait_time)
                else:
                    if self.logger:
                        log_message(self.logger, 'error',
                                   f"Failed to get dependents for {owner}/{repo} after {MAX_RETRIES + 1} attempts: {str(e)}")
                    return 0
        
        return 0


# ============================================================================
# FILE PARSING
# ============================================================================

def parse_csv_file(csv_path: str) -> List[Dict[str, str]]:
    """
    Parse a CSV file from Common-Package-Filter results.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of dictionaries with normalized_url and ecosystems
    """
    packages = []
    
    # Extract ecosystems from CSV filename
    csv_filename = os.path.basename(csv_path)
    ecosystems = csv_filename.replace(".csv", "").split("_")
    ecosystems_str = ", ".join(ecosystems)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Only use Normalized_URL column - no fallback
            normalized_url = row.get("Normalized_URL", "").strip()
            
            if not normalized_url:
                continue  # Skip rows without normalized URL
            
            # Convert normalized URL (github.com/owner/repo) to full URL and owner/repo
            # normalized_url format: github.com/owner/repo
            parts = normalized_url.replace("github.com/", "").split("/")
            if len(parts) != 2:
                continue  # Invalid format
            
            owner, repo_name = parts[0], parts[1]
            owner_repo = f"{owner}/{repo_name}"
            full_url = f"https://{normalized_url}"
            
            packages.append({
                "owner_repo": owner_repo,
                "repo_url": full_url,
                "normalized_url": normalized_url,
                "ecosystems": ecosystems_str
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
    
    # Should not reach here, but just in case
    return {"owner_repo": owner_repo, "repo_url": repo_url, "ecosystems": ecosystems, "error": "UNKNOWN", "error_detail": "No metrics returned"}


def process_packages_parallel(
    all_packages: List[Dict],
    miner: GitHubMetricsMiner,
    processed_packages: Dict,
    output_path: Path,
    max_workers: int = MAX_WORKERS,
) -> Dict[str, Dict]:
    """
    Process all packages using parallel workers.

    Args:
        all_packages: List of all packages to process
        miner: GitHubMetricsMiner instance
        processed_packages: Dictionary of already processed packages
        output_path: Path to output JSON file for saving progress
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
    save_interval = 10  # Save progress every 10 packages
    processed_count = 0
    error_count = 0
    warning_shown = False
    error_stats = {}  # Track error types and counts

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
                            # Check if result contains error
                            if "error" in result:
                                error_type = result.get("error", "UNKNOWN")
                                error_detail = result.get("error_detail", "")
                                
                                # Track error statistics
                                if error_type not in error_stats:
                                    error_stats[error_type] = 0
                                error_stats[error_type] += 1
                                
                                # HTTP 404 is expected (deleted/private repos) - don't count as critical error
                                if error_type != "HTTP_404":
                                    error_count += 1
                                
                                # Store result with error info
                                processed_packages[pkg_key] = result
                                
                                # Show warning when critical errors exceed 50
                                if error_count > 50 and not warning_shown:
                                    warning_shown = True
                                    safe_print("\n" + "!" * 80)
                                    safe_print("⚠️  WARNING: More than 50 repositories have failed to be mined!")
                                    safe_print("⚠️  Current error count: {}".format(error_count))
                                    safe_print("⚠️  Check processing.log for detailed error messages")
                                    safe_print("⚠️  Common causes: authentication issues, rate limits, network errors")
                                    safe_print("!" * 80 + "\n")
                                
                                # Continue showing periodic warnings every 100 errors after threshold
                                elif error_count > 50 and error_count % 100 == 0:
                                    safe_print("\n⚠️  Error count: {} (check processing.log)\n".format(error_count))
                            else:
                                # Success
                                processed_packages[pkg_key] = result
                        else:
                            # Should not happen with updated logic, but keep for safety
                            processed_packages[pkg_key] = {"error": "NO_RESULT", "error_detail": "No result returned"}
                            error_count += 1
                            if "NO_RESULT" not in error_stats:
                                error_stats["NO_RESULT"] = 0
                            error_stats["NO_RESULT"] += 1
                        
                        processed_count += 1
                        
                        # Save progress periodically
                        if processed_count % save_interval == 0:
                            save_progress(output_path, processed_packages, idx)
                
                except Exception as e:
                    with results_lock:
                        processed_packages[pkg_key] = {"error": "PROCESSING_EXCEPTION", "error_detail": str(e)}
                        error_count += 1
                        
                        # Track error
                        if "PROCESSING_EXCEPTION" not in error_stats:
                            error_stats["PROCESSING_EXCEPTION"] = 0
                        error_stats["PROCESSING_EXCEPTION"] += 1
                        
                        # Show warning when errors exceed 50
                        if error_count > 50 and not warning_shown:
                            warning_shown = True
                            safe_print("\n" + "!" * 80)
                            safe_print("⚠️  WARNING: More than 50 repositories have failed to be mined!")
                            safe_print("⚠️  Current error count: {}".format(error_count))
                            safe_print("⚠️  Check processing.log for detailed error messages")
                            safe_print("⚠️  Common causes: authentication issues, rate limits, network errors")
                            safe_print("!" * 80 + "\n")
                        
                        # Continue showing periodic warnings every 100 errors after threshold
                        elif error_count > 50 and error_count % 100 == 0:
                            safe_print("\n⚠️  Error count: {} (check processing.log)\n".format(error_count))
                
                pbar.update(1)

    # Save final progress
    save_progress(output_path, processed_packages, len(all_packages) - 1)
    
    return processed_packages, error_stats


def count_successful_packages(processed_packages: Dict) -> int:
    """
    Count successfully mined packages.
    
    Args:
        processed_packages: Dictionary of processed results
        
    Returns:
        Number of successfully mined packages
    """
    return len([v for v in processed_packages.values() if v and "error" not in v])


def generate_summary_stats(processed_packages: Dict, error_stats: Dict = None) -> Dict:
    """
    Generate summary statistics for the mining process.
    
    Args:
        processed_packages: Dictionary of processed results
        error_stats: Dictionary of error type counts (optional, will be computed if not provided)
        
    Returns:
        Dictionary with summary statistics
    """
    total = len(processed_packages)
    success = count_successful_packages(processed_packages)
    
    # Count errors directly from processed_packages to ensure accuracy
    computed_error_stats = {}
    forked_count = 0
    archived_count = 0
    
    for pkg_key, result in processed_packages.items():
        if result:
            if "error" in result:
                error_type = result.get("error", "UNKNOWN")
                if error_type not in computed_error_stats:
                    computed_error_stats[error_type] = 0
                computed_error_stats[error_type] += 1
            else:
                # Count forked and archived repos (only for successfully mined packages)
                if result.get("is_fork", False):
                    forked_count += 1
                if result.get("is_archived", False):
                    archived_count += 1
    
    # Use computed stats, merge with provided stats if any
    if error_stats:
        for error_type, count in error_stats.items():
            if error_type not in computed_error_stats:
                computed_error_stats[error_type] = count
    
    total_errors = sum(computed_error_stats.values())
    
    # Combined count: forked + archived + errors (for exclusion analysis)
    excluded_count = forked_count + archived_count + total_errors
    
    return {
        "total": total,
        "success": success,
        "success_rate": (success / total * 100) if total > 0 else 0,
        "total_errors": total_errors,
        "error_rate": (total_errors / total * 100) if total > 0 else 0,
        "error_breakdown": computed_error_stats,
        "forked_count": forked_count,
        "forked_rate": (forked_count / total * 100) if total > 0 else 0,
        "archived_count": archived_count,
        "archived_rate": (archived_count / total * 100) if total > 0 else 0,
        "excluded_count": excluded_count,
        "excluded_rate": (excluded_count / total * 100) if total > 0 else 0,
    }


def get_packages_with_errors(processed_packages: Dict) -> List[Dict]:
    """
    Get list of packages that have errors and should be retried.
    
    Args:
        processed_packages: Dictionary of processed results
        
    Returns:
        List of package dictionaries with errors (excluding HTTP_404 which are permanent)
    """
    packages_to_retry = []
    
    # Error types that should NOT be retried (permanent failures)
    permanent_errors = {"HTTP_404", "HTTP_451"}  # Not found, legally restricted
    
    for pkg_key, result in processed_packages.items():
        if result and "error" in result:
            error_type = result.get("error", "UNKNOWN")
            
            # Skip permanent errors
            if error_type in permanent_errors:
                continue
            
            # Extract owner_repo and repo_url from the key
            parts = pkg_key.split('|', 1)
            if len(parts) == 2:
                owner_repo, repo_url = parts
                packages_to_retry.append({
                    "owner_repo": owner_repo,
                    "repo_url": repo_url,
                    "ecosystems": result.get("ecosystems", ""),
                    "previous_error": error_type,
                    "previous_error_detail": result.get("error_detail", "")
                })
    
    return packages_to_retry


def retry_failed_packages(
    processed_packages: Dict,
    miner: 'GitHubMetricsMiner',
    output_path: Path,
    max_workers: int,
    max_retry_rounds: int,
    logger
) -> Tuple[Dict, Dict, int]:
    """
    Retry packages that have errors.
    
    Args:
        processed_packages: Dictionary of processed results
        miner: GitHubMetricsMiner instance
        output_path: Path to output JSON file
        max_workers: Number of parallel workers
        max_retry_rounds: Maximum retry rounds
        logger: Logger instance
        
    Returns:
        Tuple of (updated processed_packages, error_stats, total_retried)
    """
    total_retried = 0
    all_error_stats = {}
    
    for retry_round in range(1, max_retry_rounds + 1):
        # Get packages with errors that can be retried
        packages_to_retry = get_packages_with_errors(processed_packages)
        
        if not packages_to_retry:
            safe_print(f"\n✓ No more packages to retry")
            break
        
        safe_print(f"\n{'=' * 80}")
        safe_print(f"Retry Round {retry_round}/{max_retry_rounds}")
        safe_print(f"{'=' * 80}")
        safe_print(f"Packages to retry: {len(packages_to_retry)}")
        
        # Log packages being retried
        if logger:
            log_message(logger, 'info', f"Retry round {retry_round}: {len(packages_to_retry)} packages")
            for pkg in packages_to_retry[:10]:  # Log first 10
                log_message(logger, 'info', f"  Retrying: {pkg['owner_repo']} (previous error: {pkg['previous_error']})")
            if len(packages_to_retry) > 10:
                log_message(logger, 'info', f"  ... and {len(packages_to_retry) - 10} more")
        
        # Remove error entries from processed_packages so they can be reprocessed
        for pkg in packages_to_retry:
            pkg_key = f"{pkg['owner_repo']}|{pkg['repo_url']}"
            if pkg_key in processed_packages:
                del processed_packages[pkg_key]
        
        # Process with parallel workers
        try:
            processed_packages, error_stats = process_packages_parallel(
                packages_to_retry,
                miner,
                processed_packages,
                output_path,
                max_workers=max_workers,
            )
            
            # Merge error stats
            for error_type, count in error_stats.items():
                if error_type not in all_error_stats:
                    all_error_stats[error_type] = 0
                all_error_stats[error_type] += count
            
            total_retried += len(packages_to_retry)
            
            # Count how many were successful in this round
            successful_this_round = 0
            for pkg in packages_to_retry:
                pkg_key = f"{pkg['owner_repo']}|{pkg['repo_url']}"
                result = processed_packages.get(pkg_key)
                if result and "error" not in result:
                    successful_this_round += 1
            
            safe_print(f"\n✓ Retry round {retry_round} complete:")
            safe_print(f"  Retried: {len(packages_to_retry)}")
            safe_print(f"  Successful: {successful_this_round}")
            safe_print(f"  Still failing: {len(packages_to_retry) - successful_this_round}")
            
            if logger:
                log_message(logger, 'info', f"Retry round {retry_round} complete: {successful_this_round}/{len(packages_to_retry)} successful")
            
            # If no improvements, stop retrying
            if successful_this_round == 0:
                safe_print(f"\n⚠ No improvements in this retry round, stopping retries")
                break
                
        except KeyboardInterrupt:
            safe_print(f"\n\n⚠ Retry interrupted by user. Progress has been saved.")
            break
        except Exception as e:
            safe_print(f"\n✗ Error during retry round {retry_round}: {e}")
            if logger:
                log_message(logger, 'error', f"Retry round {retry_round} failed: {e}")
            break
    
    return processed_packages, all_error_stats, total_retried


def write_summary_file(output_dir: Path, stats: Dict, all_packages: List[Dict]):
    """
    Write summary.txt file with statistics.
    
    Args:
        output_dir: Output directory path
        stats: Summary statistics dictionary
        all_packages: List of all packages
    """
    summary_path = output_dir / "summary.txt"
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("GITHUB METRICS MINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Input Repositories: {stats['total']:,}\n")
        f.write(f"Successfully Mined: {stats['success']:,} ({stats['success_rate']:.2f}%)\n")
        f.write(f"Errors: {stats['total_errors']:,} ({stats['error_rate']:.2f}%)\n\n")
        
        # Repository status breakdown
        f.write("REPOSITORY STATUS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Forked Repositories: {stats.get('forked_count', 0):,} ({stats.get('forked_rate', 0):.2f}%)\n")
        f.write(f"Archived Repositories: {stats.get('archived_count', 0):,} ({stats.get('archived_rate', 0):.2f}%)\n")
        f.write(f"Forked + Archived + Errors: {stats.get('excluded_count', 0):,} ({stats.get('excluded_rate', 0):.2f}%)\n")
        f.write(f"  → These may need to be excluded from analysis\n\n")
        
        # Cache statistics
        if 'cache_hits' in stats:
            f.write("SOURCE BREAKDOWN\n")
            f.write("=" * 80 + "\n")
            cache_hits = stats.get('cache_hits', 0)
            api_calls = stats.get('api_calls', 0)
            total_retried = stats.get('total_retried', 0)
            total = stats['total']
            f.write(f"From Cache: {cache_hits:,} ({cache_hits / total * 100:.2f}%)\n" if total > 0 else f"From Cache: {cache_hits:,}\n")
            f.write(f"From API: {api_calls:,} ({api_calls / total * 100:.2f}%)\n" if total > 0 else f"From API: {api_calls:,}\n")
            if total_retried > 0:
                f.write(f"Retried: {total_retried:,}\n")
            f.write("\n")
        
        if stats['error_breakdown']:
            f.write("ERROR BREAKDOWN\n")
            f.write("=" * 80 + "\n")
            
            # Sort errors by count descending
            sorted_errors = sorted(stats['error_breakdown'].items(), key=lambda x: x[1], reverse=True)
            
            for error_type, count in sorted_errors:
                percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
                f.write(f"{error_type}: {count:,} ({percentage:.2f}%)\n")
                
                # Add explanation for common error types
                if error_type == "HTTP_404":
                    f.write("  → Repository not found (deleted or made private)\n")
                elif error_type == "HTTP_403":
                    f.write("  → Access forbidden (private repository or rate limit)\n")
                elif error_type == "API_REQUEST_FAILED":
                    f.write("  → Network or API connection issues\n")
                elif error_type == "EXCEPTION":
                    f.write("  → Unexpected exception during processing\n")
                f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Summary generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    safe_print(f"✓ Summary saved to {summary_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mine GitHub metrics from Common-Package-Filter CSV results (with parallel processing)"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing CSV input files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output file (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON filename (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--token",
        nargs="+",
        help="GitHub personal access token(s) (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific CSV files to process (default: all CSV files in input dir)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Directory containing cache files (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--cache-file",
        default=DEFAULT_CACHE_FILE,
        help=f"Cache JSON filename (default: {DEFAULT_CACHE_FILE})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache lookup, mine all packages via API",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable retry for packages with errors",
    )
    parser.add_argument(
        "--max-retry-rounds",
        type=int,
        default=1,
        help="Maximum number of retry rounds for failed packages (default: 1)",
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

    # Setup logger
    logger = setup_logging()
    token_manager.logger = logger
    
    # Initialize miner
    miner = GitHubMetricsMiner(token_manager, logger)

    # Get input directory
    input_dir = (script_dir / args.input_dir).resolve()
    print(f"Input directory: {input_dir}")
    
    # Output file path
    output_path = output_dir / args.output_file
    print(f"Output file: {output_path}")

    # Get CSV files to process
    csv_files = []
    if args.files:
        # If specific files provided
        for f in args.files:
            file_path = input_dir / f
            if file_path.is_file():
                csv_files.append(file_path)
            else:
                # Try searching in subdirectories
                for subdir in input_dir.iterdir():
                    if subdir.is_dir():
                        possible_path = subdir / f
                        if possible_path.is_file():
                            csv_files.append(possible_path)
                            break
    else:
        # Get all CSV files from subdirectories
        ecosystem_dirs = [
            d
            for d in input_dir.iterdir()
            if d.is_dir() and d.name.endswith("_ecosystems")
        ]

        if ecosystem_dirs:
            # Structure with ecosystem folders
            for ecosystem_dir in sorted(ecosystem_dirs):
                csv_files.extend(list(ecosystem_dir.glob("*.csv")))
        else:
            # Fallback: CSV files directly in input_dir
            csv_files = list(input_dir.glob("*.csv"))

    csv_files = [f for f in csv_files if f.is_file()]

    if not csv_files:
        print(f"\n✗ No CSV files found in {input_dir}")
        return

    print(f"\nFound {len(csv_files)} CSV file(s) to process:")

    # Group by ecosystem count for display
    files_by_count = {}
    for f in csv_files:
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

    # Load progress if exists
    output_path = output_dir / args.output_file
    processed_packages, _ = load_progress(output_path)
    if processed_packages:
        successful = count_successful_packages(processed_packages)
        safe_print(f"\nResuming from previous progress: {successful}/{len(processed_packages)} packages successfully processed")

    # Collect all packages from all files
    safe_print(f"\n{'=' * 80}")
    safe_print("Parsing input files...")
    safe_print(f"{'=' * 80}")
    
    all_packages = []
    files_by_count_stats = {}
    
    for csv_file in tqdm(csv_files, desc="Parsing files", unit="file"):
        try:
            packages = parse_csv_file(str(csv_file))
            all_packages.extend(packages)
            
            # Track stats by ecosystem count
            parent = csv_file.parent.name
            if parent.endswith("_ecosystems"):
                count = int(parent.split("_")[0])
            else:
                count = len(csv_file.stem.split("_"))
            
            if count not in files_by_count_stats:
                files_by_count_stats[count] = {"files": 0, "packages": 0}
            files_by_count_stats[count]["files"] += 1
            files_by_count_stats[count]["packages"] += len(packages)
            
        except Exception as e:
            safe_print(f"\n✗ Error parsing {csv_file.name}: {e}")
            continue

    # Remove duplicates (same owner/repo)
    unique_packages = {}
    for pkg in all_packages:
        key = pkg["owner_repo"]
        if key not in unique_packages:
            unique_packages[key] = pkg
    
    all_packages = list(unique_packages.values())
    
    safe_print(f"\nTotal unique repositories: {len(all_packages)}")
    safe_print(f"Files parsed: {len(csv_files)}")
    
    # Display stats by ecosystem count
    safe_print(f"\nPackages by ecosystem count:")
    for count in sorted(files_by_count_stats.keys()):
        stats = files_by_count_stats[count]
        safe_print(f"  {count}-ecosystems: {stats['packages']} packages from {stats['files']} files")

    # ========================================================================
    # CACHE LOADING AND TRANSFER
    # ========================================================================
    cache = {}
    cache_index = {}
    cache_hits = 0
    remaining_packages = all_packages
    
    if not args.no_cache:
        safe_print(f"\n{'=' * 80}")
        safe_print("Loading and indexing cache...")
        safe_print(f"{'=' * 80}")
        
        # Resolve cache path
        cache_dir = (script_dir / args.cache_dir).resolve()
        cache_path = cache_dir / args.cache_file
        
        safe_print(f"Cache directory: {cache_dir}")
        safe_print(f"Cache file: {cache_path}")
        
        # Load cache
        cache = load_cache(cache_path)
        
        if cache:
            # Build search index
            safe_print("\nBuilding cache index...")
            cache_index = build_cache_index(cache)
            safe_print(f"  Index contains {len(cache_index)} lookup keys")
            
            # Transfer cached packages
            safe_print(f"\n{'=' * 80}")
            safe_print("Transferring cached packages...")
            safe_print(f"{'=' * 80}")
            
            processed_packages, remaining_packages, cache_hits = transfer_from_cache(
                all_packages, cache, cache_index, processed_packages
            )
            
            safe_print(f"\n✓ Cache lookup complete:")
            safe_print(f"  Total packages to process: {len(all_packages)}")
            safe_print(f"  Found in cache: {cache_hits} ({cache_hits / len(all_packages) * 100:.2f}%)")
            safe_print(f"  Remaining to mine via API: {len(remaining_packages)} ({len(remaining_packages) / len(all_packages) * 100:.2f}%)")
            
            # Save progress after cache transfer
            if cache_hits > 0:
                save_progress(output_path, processed_packages, cache_hits - 1)
                safe_print(f"  Progress saved after cache transfer")
        else:
            safe_print("\nNo valid cache found, will mine all packages via API")
    else:
        safe_print(f"\n⚠ Cache disabled (--no-cache flag), will mine all packages via API")

    # Process remaining packages via API
    error_stats = {}
    
    if remaining_packages:
        try:
            safe_print(f"\n{'=' * 80}")
            safe_print("Mining GitHub metrics via API...")
            safe_print(f"{'=' * 80}\n")
            
            processed_packages, error_stats = process_packages_parallel(
                remaining_packages,
                miner,
                processed_packages,
                output_path,
                max_workers=args.workers,
            )
            
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user. Progress has been saved.")
            print("Run the script again to resume from checkpoint.")
            return
        except Exception as e:
            safe_print(f"\n✗ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        safe_print(f"\n✓ All packages found in cache, no API calls needed!")
        # Save final state
        save_progress(output_path, processed_packages, len(all_packages) - 1)
    
    # ========================================================================
    # RETRY PHASE: Retry packages with errors
    # ========================================================================
    total_retried = 0
    
    if not args.no_retry:
        # Check if there are packages with retryable errors
        packages_with_errors = get_packages_with_errors(processed_packages)
        
        if packages_with_errors:
            safe_print(f"\n{'=' * 80}")
            safe_print(f"Found {len(packages_with_errors)} packages with retryable errors")
            safe_print(f"{'=' * 80}")
            
            # Log the error types
            error_type_counts = {}
            for pkg in packages_with_errors:
                error_type = pkg.get('previous_error', 'UNKNOWN')
                if error_type not in error_type_counts:
                    error_type_counts[error_type] = 0
                error_type_counts[error_type] += 1
            
            safe_print("Error types to retry:")
            for error_type, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True):
                safe_print(f"  • {error_type}: {count}")
            
            # Perform retries
            processed_packages, retry_error_stats, total_retried = retry_failed_packages(
                processed_packages,
                miner,
                output_path,
                max_workers=args.workers,
                max_retry_rounds=args.max_retry_rounds,
                logger=logger
            )
            
            # Merge retry error stats
            for error_type, count in retry_error_stats.items():
                if error_type not in error_stats:
                    error_stats[error_type] = 0
                error_stats[error_type] += count
        else:
            safe_print(f"\n✓ No packages with retryable errors")
    else:
        safe_print(f"\n⚠ Retry disabled (--no-retry flag)")
    
    successful_count = count_successful_packages(processed_packages)
    
    # Generate summary statistics (this computes error stats from processed_packages)
    summary_stats = generate_summary_stats(processed_packages)
    
    # Add cache stats to summary
    summary_stats['cache_hits'] = cache_hits
    summary_stats['api_calls'] = len(remaining_packages)
    summary_stats['total_retried'] = total_retried
    
    safe_print(f"\n✓ Results saved to {output_path}")
    safe_print(f"  Successfully mined: {successful_count}/{len(all_packages)} repositories")
    
    # Write summary file
    write_summary_file(output_dir, summary_stats, all_packages)

    # Summary
    print(f"\n{'=' * 80}")
    print("Processing Complete!")
    print(f"{'=' * 80}")
    print(f"Output file: {output_path}")
    print(f"Summary file: {output_dir / 'summary.txt'}")
    print(f"\nTotal repositories processed: {len(all_packages)}")
    print(f"  From cache: {cache_hits} ({cache_hits / len(all_packages) * 100:.2f}%)" if len(all_packages) > 0 else "  From cache: 0")
    print(f"  From API: {len(remaining_packages)} ({len(remaining_packages) / len(all_packages) * 100:.2f}%)" if len(all_packages) > 0 else "  From API: 0")
    if total_retried > 0:
        print(f"  Retried: {total_retried}")
    print(f"Successfully mined: {successful_count} ({summary_stats['success_rate']:.2f}%)")
    print(f"Errors: {summary_stats['total_errors']} ({summary_stats['error_rate']:.2f}%)")
    
    if summary_stats['error_breakdown']:
        print(f"\nTop error types:")
        sorted_errors = sorted(summary_stats['error_breakdown'].items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors[:5]:  # Show top 5
            percentage = (count / len(all_packages) * 100) if len(all_packages) > 0 else 0
            print(f"  • {error_type}: {count} ({percentage:.2f}%)")
    
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
