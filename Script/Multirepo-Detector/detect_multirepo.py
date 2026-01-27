import os
import re
import csv
import time
import requests
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from requests.exceptions import RequestException, ConnectionError, Timeout
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging

# Define paths
DIRECTORY_STRUCTURE_PATH = Path(__file__).parent / "../../Resource/Dataset/Directory-Structure-Miner"
OUTPUT_DIR = Path(__file__).parent / "../../Resource/Dataset/Multirepo-Detector"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "processing.log"

# Thread-safe printing and logging
print_lock = Lock()
log_lock = Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)

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


class TokenManager:
    """Manages GitHub API tokens with rotation and validation."""
    
    def __init__(self, tokens: List[str] = None):
        """Initialize token manager with a list of tokens."""
        self.tokens = tokens or []
        self.current_token_index = 0
        self.base_url = "https://api.github.com"
        self.lock = Lock()  # Thread-safe token rotation
        self.repo_cache: Dict[str, List[str]] = {}  # Cache for user repos
        self.cache_lock = Lock()  # Thread-safe cache access
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
                    core = data.get('rate', {})
                    remaining = core.get('remaining', 'N/A')
                    reset_timestamp = core.get('reset', 0)
                    reset_time = datetime.fromtimestamp(reset_timestamp).strftime('%H:%M:%S')
                    status = "Valid"
                elif response.status_code == 401:
                    status = "Invalid"
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
                core = data.get('rate', {})
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
    
    def check_rate_limit(self, response: requests.Response) -> bool:
        """Check rate limit from response and rotate token if needed. Returns True if rotated."""
        remaining = response.headers.get('X-RateLimit-Remaining')
        
        if remaining and int(remaining) < 5:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            
            # Try to rotate to another token
            if self.rotate_token():
                return True
            else:
                # No other tokens, need to wait
                wait_time = max(reset_time - time.time(), 0) + 5
                print(f"  Rate limit reached. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
                return False
        
        return False


# ==============================================================================
# ECOSYSTEM SUFFIX PATTERNS (Expanded and Organized)
# ==============================================================================

# Ecosystem suffixes organized by package manager/ecosystem
# These are used to detect multi-repo patterns like: project-python, project-rust, etc.
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
        # Additional languages/ecosystems
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

# Flatten all suffixes into a single list for backward compatibility
ECOSYSTEM_SUFFIXES = []
for ecosystem, suffixes in ECOSYSTEM_SUFFIX_PATTERNS.items():
    ECOSYSTEM_SUFFIXES.extend(suffixes)

# Remove duplicates while preserving order
seen = set()
ECOSYSTEM_SUFFIXES = [x for x in ECOSYSTEM_SUFFIXES if not (x in seen or seen.add(x))]

# Valid GitHub repo name separators
SEPARATORS = ['-', '_', '.']

# Rate limiting
RATE_LIMIT_DELAY = 1  # seconds between API calls

# Retry configuration
MAX_RETRIES = 3  # Maximum number of retry attempts for network errors
RETRY_DELAY = 2  # Initial delay in seconds before retry (will use exponential backoff)

# Parallel processing configuration
MAX_WORKERS = 10  # Number of concurrent threads for API calls

def parse_github_url(url):
    """
    Parse GitHub URL to extract owner and repo name.
    
    Args:
        url: GitHub repository URL
    
    Returns:
        Tuple of (owner, repo) or (None, None) if parsing fails
    """
    if not url or not str(url).strip():
        return None, None
    
    # Convert to string and normalize
    url = str(url).strip().lower()
    
    # Check if it's a GitHub URL
    if "github.com" not in url:
        return None, None
    
    # Remove common suffixes and prefixes
    url = re.sub(r"\.git$", "", url)
    url = re.sub(r"/$", "", url)
    
    # Remove git protocol prefixes
    url = url.replace('git+https://', 'https://')
    url = url.replace('git+ssh://', 'ssh://')
    url = url.replace('git://', 'https://')
    
    # Extract path from URL using regex to handle both : and / separators
    try:
        # Handle various GitHub URL formats (https, ssh, git@)
        match = re.search(r"github\.com[:/]([^/]+/[^/\s]+)", url)
        if match:
            repo_path = match.group(1)
            # Remove trailing content after repository name
            repo_path = re.split(r"[\s#?]", repo_path)[0]
            # Remove .git suffix if still present in the extracted path
            repo_path = re.sub(r"\.git$", "", repo_path)
            # Remove trailing slash
            repo_path = repo_path.rstrip('/')
            
            # Split into owner and repo
            path_parts = repo_path.split("/")
            if len(path_parts) >= 2:
                owner = path_parts[0]
                repo = path_parts[1]
                return owner, repo
    except Exception:
        pass
    
    return None, None


def check_if_forked_or_archived(owner, repo, token_manager: TokenManager, logger=None):
    """
    Check if a repository is forked from another repository or archived.
    Returns: (should_skip, reason_message)
        - should_skip: True if repo is forked or archived
        - reason_message: Description of why it should be skipped, or None
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    
    # Ensure we have rate limit available
    token = token_manager.ensure_rate_limit()
    headers = {'Authorization': f'token {token}'} if token else {}
    
    retry_count = 0
    while retry_count <= MAX_RETRIES:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 404:
                error_msg = f"Repository {owner}/{repo} not found (404)"
                if logger:
                    log_message(logger, 'warning', error_msg)
                return False, error_msg
            
            if response.status_code == 403:
                # Rate limit, ensure we have available limit
                token = token_manager.ensure_rate_limit()
                headers = {'Authorization': f'token {token}'} if token else {}
                retry_count += 1
                continue
            
            if response.status_code != 200:
                error_msg = f"Failed to check repo status for {owner}/{repo}: HTTP {response.status_code}"
                if logger:
                    log_message(logger, 'error', error_msg)
                return False, error_msg
            
            repo_data = response.json()
            
            # Check if archived
            is_archived = repo_data.get('archived', False)
            if is_archived:
                return True, 'archived'
            
            # Check if fork
            is_fork = repo_data.get('fork', False)
            if is_fork:
                parent = repo_data.get('parent', {})
                parent_url = parent.get('html_url', 'Unknown parent')
                return True, f'forked from {parent_url}'
            
            return False, None
            
        except (ConnectionError, Timeout, RequestException) as e:
            retry_count += 1
            if retry_count <= MAX_RETRIES:
                delay = RETRY_DELAY * (2 ** (retry_count - 1))
                time.sleep(delay)
            else:
                error_msg = f"Network error checking repo status for {owner}/{repo}: {type(e).__name__}: {str(e)}"
                if logger:
                    log_message(logger, 'error', error_msg)
                return False, error_msg
        
        except Exception as e:
            error_msg = f"Unexpected error checking repo status for {owner}/{repo}: {type(e).__name__}: {str(e)}"
            if logger:
                log_message(logger, 'error', error_msg)
            return False, error_msg
    
    error_msg = f"Max retries exceeded checking repo status for {owner}/{repo}"
    if logger:
        log_message(logger, 'error', error_msg)
    return False, error_msg


def get_user_repos(owner, token_manager: TokenManager):
    """
    Fetch all public repositories for a GitHub user/organization.
    Returns list of repository names.
    Implements retry logic for transient network errors.
    Uses caching to avoid redundant API calls.
    """
    # Check cache first
    with token_manager.cache_lock:
        if owner in token_manager.repo_cache:
            return token_manager.repo_cache[owner]
    
    all_repos = []
    page = 1
    per_page = 100
    
    # Ensure we have rate limit available
    token = token_manager.ensure_rate_limit()
    headers = {'Authorization': f'token {token}'} if token else {}
    
    while True:
        url = f"https://api.github.com/users/{owner}/repos"
        params = {'per_page': per_page, 'page': page}
        
        # Retry loop for transient network errors
        retry_count = 0
        success = False
        
        while retry_count <= MAX_RETRIES:
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                if response.status_code == 404:
                    # User/org not found, might be private or doesn't exist
                    with token_manager.cache_lock:
                        token_manager.repo_cache[owner] = []
                    return []
                
                if response.status_code != 200:
                    if response.status_code == 403:  # Rate limit
                        # Ensure rate limit and get a fresh token
                        token = token_manager.ensure_rate_limit()
                        headers = {'Authorization': f'token {token}'} if token else {}
                        retry_count += 1
                        continue
                    with token_manager.cache_lock:
                        token_manager.repo_cache[owner] = []
                    return []
                
                repos = response.json()
                
                if not repos:  # No more repos
                    success = True
                    break
                
                all_repos.extend([repo['name'] for repo in repos if isinstance(repo, dict)])
                
                if len(repos) < per_page:  # Last page
                    success = True
                    break
                
                page += 1
                
                # Success - break out of retry loop
                success = True
                break
                
            except (ConnectionError, Timeout, RequestException) as e:
                retry_count += 1
                
                if retry_count <= MAX_RETRIES:
                    delay = RETRY_DELAY * (2 ** (retry_count - 1))  # Exponential backoff
                    time.sleep(delay)
                else:
                    with token_manager.cache_lock:
                        token_manager.repo_cache[owner] = []
                    return []
                    
            except Exception as e:
                with token_manager.cache_lock:
                    token_manager.repo_cache[owner] = []
                return []
        
        # If we successfully processed the page, continue to next or break
        if success:
            if not repos or len(repos) < per_page:
                break
        else:
            with token_manager.cache_lock:
                token_manager.repo_cache[owner] = []
            return []
    
    # Cache the results
    with token_manager.cache_lock:
        token_manager.repo_cache[owner] = all_repos
    
    return all_repos


def detect_multirepo(owner, repo_name, token_manager: TokenManager):
    """
    Detect if a repository follows multi-repo pattern.
    Returns: (is_multirepo, separator, found_repos)
    Implements retry logic for transient network errors.
    
    New logic to handle mixed separators:
    1. Add repo name itself as a candidate
    2. Try removing <separator><suffix> from end or <suffix><separator> from start
       (checking ALL separators)
    3. Use each base candidate to construct patterns with ALL separators
    """
    # Get all repos from the owner (with retry logic built-in)
    all_repos = get_user_repos(owner, token_manager)
    
    if not all_repos:
        return False, None, []
    
    # Step 1: Add repo name as a candidate
    base_candidates = [repo_name]
    
    # Step 2: Check for ending <separator><suffix> and starting <suffix><separator>
    # Try ALL separators to handle mixed separator cases
    for separator in SEPARATORS:
        for suffix in ECOSYSTEM_SUFFIXES:
            # Check ending: <base><separator><suffix>
            if repo_name.endswith(f"{separator}{suffix}"):
                base = repo_name[:-(len(separator) + len(suffix))]
                if base:  # Only add non-empty bases
                    base_candidates.append(base)
            
            # Check starting: <suffix><separator><base>
            if repo_name.startswith(f"{suffix}{separator}"):
                base = repo_name[(len(suffix) + len(separator)):]
                if base:  # Only add non-empty bases
                    base_candidates.append(base)
    
    # Remove duplicates while preserving order
    seen = set()
    base_candidates = [x for x in base_candidates if not (x in seen or seen.add(x))]
    
    # Step 3: For each base candidate, try constructing patterns with ALL separators
    for base_name in base_candidates:
        if not base_name:
            continue
        
        for separator in SEPARATORS:
            # Look for pattern: <base><separator><ecosystem>
            multi_repo_patterns = [f"{base_name}{separator}{suffix}" for suffix in ECOSYSTEM_SUFFIXES]
            found_multi_repos = [r for r in all_repos if r in multi_repo_patterns]
            
            # If we found at least 2 repos matching the pattern, it's a multi-repo
            if len(found_multi_repos) >= 2:
                return True, separator, sorted(found_multi_repos)
    
    return False, None, []


def parse_directory_structure_file(file_path):
    """
    Parse a directory structure file and extract package information.
    Returns list of packages with their details.
    """
    packages = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by package sections
    # Pattern: "Package N/M" followed by "===" line
    package_pattern = re.compile(
        r'={80,}\n'
        r'Package \d+/\d+\n'
        r'={80,}\n'
        r'(.*?)'
        r'(?=\n={80,}\nPackage \d+/\d+\n={80,}|\Z)',
        re.DOTALL
    )
    
    matches = package_pattern.findall(content)
    
    for match in matches:
        lines = match.strip().split('\n')
        
        package_info = {
            'repo_url': '',
            'owner_repo': '',
            'ecosystems': '',
            'full_text': match.strip()
        }
        
        for line in lines[:10]:  # Check first 10 lines for metadata
            if line.startswith('Repository:'):
                package_info['repo_url'] = line.split('Repository:', 1)[1].strip()
            elif line.startswith('Owner/Repo:'):
                package_info['owner_repo'] = line.split('Owner/Repo:', 1)[1].strip()
            elif line.startswith('Ecosystems:'):
                package_info['ecosystems'] = line.split('Ecosystems:', 1)[1].strip()
        
        if package_info['repo_url']:
            packages.append(package_info)
    
    return packages


def process_single_package(package, idx, total, token_manager: TokenManager, logger=None):
    """
    Process a single package to detect if it's a multi-repo.
    Returns tuple of (package, is_multirepo, result_dict)
    """
    try:
        repo_url = package.get('repo_url', '')
        if not repo_url:
            error_msg = f"Missing repo_url in package dict"
            if logger:
                log_message(logger, 'error', error_msg)
            return (package, False, {
                'error': True,
                'message': error_msg
            })
        
        owner, repo = parse_github_url(repo_url)
        
        if not owner or not repo:
            error_msg = f"Could not parse owner/repo from URL: {repo_url} (owner={owner}, repo={repo})"
            if logger:
                log_message(logger, 'error', error_msg)
            return (package, False, {
                'error': True,
                'message': error_msg
            })
    except KeyError as e:
        error_msg = f"KeyError accessing package data: {e}"
        if logger:
            log_message(logger, 'error', error_msg)
        return (package, False, {
            'error': True,
            'message': error_msg
        })
    except Exception as e:
        error_msg = f"Error in URL parsing: {type(e).__name__}: {e}"
        if logger:
            log_message(logger, 'error', error_msg)
        return (package, False, {
            'error': True,
            'message': error_msg
        })
    
    # Check if repository is forked or archived
    try:
        should_skip, reason = check_if_forked_or_archived(owner, repo, token_manager, logger)
        
        if should_skip:
            skip_msg = f"Skipped {owner}/{repo}: {reason}"
            if logger:
                log_message(logger, 'info', skip_msg)
            return (package, False, {
                'skipped': True,
                'message': reason
            })
    except Exception as e:
        error_msg = f"Error checking repo status for {owner}/{repo}: {type(e).__name__}: {str(e)}"
        if logger:
            log_message(logger, 'error', error_msg)
        return (package, False, {
            'error': True,
            'message': error_msg
        })
    
    try:
        is_multirepo, separator, found_repos = detect_multirepo(owner, repo, token_manager)
        
        if is_multirepo:
            if logger:
                log_message(logger, 'info', f"Multi-repo detected: {owner}/{repo} (separator: {separator}, repos: {len(found_repos)})")
            return (package, True, {
                'name': repo,
                'root_url': f"https://github.com/{owner}/{repo}",
                'ecosystems': package.get('ecosystems', ''),
                'separator': separator,
                'found_repos': found_repos
            })
        else:
            return (package, False, {'message': 'mono-repo'})
    
    except Exception as e:
        error_msg = f"Error processing {owner}/{repo}: {type(e).__name__}: {str(e)}"
        if logger:
            log_message(logger, 'error', error_msg)
        return (package, False, {
            'error': True,
            'message': error_msg
        })


def process_all_files(token_manager: TokenManager, logger):
    """
    Process all directory structure files and detect multi-repo packages.
    """
    multirepo_packages = []
    statistics = {
        'total_packages': 0,
        'multirepo_count': 0,
        'monorepo_count': 0,
        'error_count': 0,
        'skipped_forks': 0,
        'skipped_archived': 0,
        'files_processed': [],
        'ecosystem_counts': {}  # Track stats by ecosystem count
    }
    
    log_message(logger, 'info', f"Starting multi-repo detection process")
    log_message(logger, 'info', f"Source directory: {DIRECTORY_STRUCTURE_PATH.absolute()}")
    log_message(logger, 'info', f"Output directory: {OUTPUT_DIR.absolute()}")
    
    print("=" * 80)
    print("Multi-Repo Detector")
    print("=" * 80)
    print()
    
    # Process each ecosystem count directory
    for ecosystem_count in range(2, 8):
        ecosystem_dir = DIRECTORY_STRUCTURE_PATH / f"{ecosystem_count}_ecosystems"
        
        if not ecosystem_dir.exists():
            continue
        
        # Initialize stats for this ecosystem count
        statistics['ecosystem_counts'][ecosystem_count] = {
            'total': 0,
            'multirepo': 0,
            'monorepo': 0,
            'skipped_forks': 0,
            'skipped_archived': 0,
            'errors': 0
        }
        
        print(f"\nProcessing {ecosystem_count}_ecosystems directory...")
        
        # Create output directory
        output_subdir = OUTPUT_DIR / f"{ecosystem_count}_ecosystems"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        for txt_file in sorted(ecosystem_dir.glob("*.txt")):
            print(f"  Reading {txt_file.name}...")
            
            packages = parse_directory_structure_file(txt_file)
            statistics['total_packages'] += len(packages)
            statistics['ecosystem_counts'][ecosystem_count]['total'] += len(packages)
            
            monorepo_packages = []
            file_multirepo_count = 0
            file_error_count = 0
            file_skipped_forks = 0
            file_skipped_archived = 0
            
            # Process packages in parallel
            print(f"    Processing {len(packages)} packages in parallel (max {MAX_WORKERS} workers)...")
            log_message(logger, 'info', f"Processing file: {txt_file.name} ({len(packages)} packages)")
            
            completed = 0
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_single_package, package, idx, len(packages), token_manager, logger): idx
                    for idx, package in enumerate(packages, 1)
                }
                
                # Process completed tasks
                for future in as_completed(futures):
                    idx = futures[future]
                    completed += 1
                    
                    try:
                        package, is_multirepo, result = future.result()
                        
                        owner, repo = parse_github_url(package['repo_url'])
                        repo_name = f"{owner}/{repo}" if owner and repo else "unknown"
                        
                        if result.get('error'):
                            safe_print(f"    [{completed}/{len(packages)}] {repo_name}: ERROR - {result['message']}")
                            file_error_count += 1
                            monorepo_packages.append(package)
                            statistics['error_count'] += 1
                            statistics['ecosystem_counts'][ecosystem_count]['errors'] += 1
                        elif result.get('skipped'):
                            skip_reason = result['message']
                            if 'archived' in skip_reason:
                                safe_print(f"    [{completed}/{len(packages)}] {repo_name}: SKIPPED (archived)")
                                file_skipped_archived += 1
                                statistics['skipped_archived'] += 1
                                statistics['ecosystem_counts'][ecosystem_count]['skipped_archived'] += 1
                            elif 'forked' in skip_reason:
                                safe_print(f"    [{completed}/{len(packages)}] {repo_name}: SKIPPED (fork)")
                                file_skipped_forks += 1
                                statistics['skipped_forks'] += 1
                                statistics['ecosystem_counts'][ecosystem_count]['skipped_forks'] += 1
                            else:
                                safe_print(f"    [{completed}/{len(packages)}] {repo_name}: SKIPPED ({skip_reason})")
                                file_skipped_forks += 1
                                statistics['skipped_forks'] += 1
                                statistics['ecosystem_counts'][ecosystem_count]['skipped_forks'] += 1
                        elif is_multirepo:
                            safe_print(f"    [{completed}/{len(packages)}] {repo_name}: MULTI-REPO "
                                      f"(sep: {result['separator']}, repos: {len(result['found_repos'])})")
                            multirepo_packages.append(result)
                            file_multirepo_count += 1
                            statistics['multirepo_count'] += 1
                            statistics['ecosystem_counts'][ecosystem_count]['multirepo'] += 1
                        else:
                            safe_print(f"    [{completed}/{len(packages)}] {repo_name}: mono-repo")
                            monorepo_packages.append(package)
                            statistics['monorepo_count'] += 1
                            statistics['ecosystem_counts'][ecosystem_count]['monorepo'] += 1
                    
                    except Exception as e:
                        safe_print(f"    [{completed}/{len(packages)}] Future failed: {e}")
                        file_error_count += 1
                        statistics['error_count'] += 1
            
            # Write filtered results (only mono-repo packages)
            output_file = output_subdir / txt_file.name
            write_filtered_results(output_file, txt_file, monorepo_packages)
            
            log_message(logger, 'info', 
                       f"Completed {txt_file.name}: {file_multirepo_count} multi-repo, "
                       f"{len(monorepo_packages) - file_error_count} mono-repo, "
                       f"{file_skipped_forks} skipped (forks), {file_skipped_archived} skipped (archived), {file_error_count} errors")
            
            statistics['files_processed'].append({
                'file': txt_file.name,
                'total': len(packages),
                'multirepo': file_multirepo_count,
                'monorepo': len(monorepo_packages) - file_error_count,
                'skipped_forks': file_skipped_forks,
                'skipped_archived': file_skipped_archived,
                'errors': file_error_count
            })
    
    # Write multi-repo packages to CSV
    write_multirepo_csv(multirepo_packages)
    
    # Generate summary
    generate_summary(statistics)
    
    return statistics


def write_filtered_results(output_file, original_file, monorepo_packages):
    """
    Write filtered results (mono-repo packages only) to output file.
    Maintains same format as input file.
    """
    with open(original_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Find header (everything before first Package section)
    header_lines = []
    for i, line in enumerate(lines):
        if re.match(r'^={80,}$', line) and i + 1 < len(lines) and lines[i + 1].startswith('Package '):
            header_lines = lines[:i]
            break
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.writelines(header_lines)
        
        # Update counts in header
        f.seek(0)
        header_content = ''.join(header_lines)
        # Update total packages count
        header_content = re.sub(
            r'Successfully Mined: \d+',
            f'Successfully Mined: {len(monorepo_packages)}',
            header_content
        )
        f.write(header_content)
        
        # Write each mono-repo package
        for idx, package in enumerate(monorepo_packages, 1):
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"Package {idx}/{len(monorepo_packages)}\n")
            f.write("=" * 80 + "\n")
            f.write("\n")
            f.write(package['full_text'])
            f.write("\n\n")


def write_multirepo_csv(multirepo_packages):
    """
    Write multi-repo packages to CSV file.
    """
    csv_file = OUTPUT_DIR / "multirepo_packages.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['name', 'root_url', 'ecosystems', 'separator', 'found_repos']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for package in multirepo_packages:
            writer.writerow({
                'name': package['name'],
                'root_url': package['root_url'],
                'ecosystems': package['ecosystems'],
                'separator': package['separator'],
                'found_repos': ', '.join(package['found_repos'])
            })
    
    print(f"\nMulti-repo packages saved to: {csv_file}")


def generate_summary(statistics):
    """
    Generate a summary file with statistics.
    """
    summary_file = OUTPUT_DIR / "summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Multi-Repo Detection Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Overall Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Packages Analyzed: {statistics['total_packages']}\n")
        f.write(f"Multi-Repo Packages: {statistics['multirepo_count']} "
                f"({statistics['multirepo_count']/max(statistics['total_packages'], 1)*100:.2f}%)\n")
        f.write(f"Mono-Repo Packages: {statistics['monorepo_count']} "
                f"({statistics['monorepo_count']/max(statistics['total_packages'], 1)*100:.2f}%)\n")
        f.write(f"Skipped (Forked): {statistics['skipped_forks']} "
                f"({statistics['skipped_forks']/max(statistics['total_packages'], 1)*100:.2f}%)\n")
        f.write(f"Skipped (Archived): {statistics['skipped_archived']} "
                f"({statistics['skipped_archived']/max(statistics['total_packages'], 1)*100:.2f}%)\n")
        f.write(f"Errors: {statistics['error_count']}\n\n")
        
        # Add ecosystem count summary table
        f.write("Statistics by Ecosystem Count\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Ecosystem Count':<18} {'Total':<10} {'Multi':<10} {'Mono':<10} {'Forks':<10} {'Archived':<12} {'Errors':<10}\n")
        f.write("-" * 80 + "\n")
        
        for count in sorted(statistics['ecosystem_counts'].keys()):
            stats = statistics['ecosystem_counts'][count]
            f.write(f"{f'{count}_ecosystems':<18} {stats['total']:<10} "
                    f"{stats['multirepo']:<10} {stats['monorepo']:<10} "
                    f"{stats['skipped_forks']:<10} {stats['skipped_archived']:<12} {stats['errors']:<10}\n")
        
        f.write("\n")
        
        f.write("File Processing Details\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'File':<40} {'Total':<8} {'Multi':<8} {'Mono':<8} {'Forks':<8} {'Archived':<10} {'Errors':<8}\n")
        f.write("-" * 80 + "\n")
        
        for file_info in statistics['files_processed']:
            f.write(f"{file_info['file']:<40} {file_info['total']:<8} "
                    f"{file_info['multirepo']:<8} {file_info['monorepo']:<8} "
                    f"{file_info['skipped_forks']:<8} {file_info['skipped_archived']:<10} {file_info['errors']:<8}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Output Files\n")
        f.write("-" * 80 + "\n")
        f.write(f"Results directory: {OUTPUT_DIR.absolute()}\n")
        f.write(f"Multi-repo CSV: multirepo_packages.csv\n")
        f.write(f"Processing log: processing.log\n")
        f.write(f"Filtered directory structures: Organized in ecosystem count subdirectories\n")
    
    print(f"\nSummary saved to: {summary_file}")


def main():
    """
    Main function to orchestrate the multi-repo detection process.
    """
    start_time = time.time()
    
    # Setup logging
    logger = setup_logging()
    
    print("=" * 80)
    print("Multi-Repo Detector (Parallel Processing)")
    print("=" * 80)
    print()
    
    log_message(logger, 'info', "Multi-Repo Detector started")
    log_message(logger, 'info', f"Log file: {LOG_FILE.absolute()}")
    
    # Prompt for GitHub tokens
    print("GitHub Token Configuration")
    print("-" * 80)
    print("You can provide one or more GitHub tokens for authentication.")
    print("Multiple tokens can be separated by comma (,) or space.")
    print("Leave empty to use unauthenticated API (60 requests/hour).")
    print()
    
    # Check for environment variable first
    env_token = os.environ.get('GITHUB_TOKEN', '').strip()
    if env_token:
        print(f"Found GITHUB_TOKEN in environment: {env_token[:8]}...")
        use_env = input("Use this token? (yes/no): ").strip().lower()
        if use_env in ['yes', 'y']:
            token_input = env_token
        else:
            token_input = input("Enter GitHub token(s) [or press Enter to skip]: ").strip()
    else:
        token_input = input("Enter GitHub token(s) [or press Enter to skip]: ").strip()
    
    # Parse tokens
    tokens = []
    if token_input:
        # Split by comma or space
        if ',' in token_input:
            tokens = [t.strip() for t in token_input.split(',') if t.strip()]
        else:
            tokens = [t.strip() for t in token_input.split() if t.strip()]
    
    # Initialize token manager
    token_manager = TokenManager(tokens)
    token_manager.logger = logger  # Set logger for rate limit messages
    
    print()
    if tokens:
        print(f"✓ Loaded {len(tokens)} token(s)")
        print("Using authenticated GitHub API (5,000 requests/hour per token)")
        
        # Validate tokens
        token_manager.validate_tokens()
        
        # Ask user to proceed
        try:
            response = input("Do you want to proceed with these tokens? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("\n✗ Operation cancelled by user.")
                return
        except (KeyboardInterrupt, EOFError):
            print("\n\n✗ Operation cancelled by user.")
            return
    else:
        print("⚠  WARNING: No GitHub Token Provided")
        print("-" * 80)
        print("Running without authentication has severe limitations:")
        print("  • Rate limit: Only 60 requests per hour (vs 5,000 with token)")
        print("  • Cannot access private repositories")
        print("  • May encounter frequent rate limit errors")
        print("  • Processing will be significantly slower")
        print()
        print("Get a token at: https://github.com/settings/tokens")
        print("-" * 80)
        
        try:
            response = input("\nDo you want to continue without a token? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("\n✗ Operation cancelled by user.")
                return
            print("\n✓ Continuing without token...")
        except (KeyboardInterrupt, EOFError):
            print("\n\n✗ Operation cancelled by user.")
            return
    
    print()
    print(f"Source directory: {DIRECTORY_STRUCTURE_PATH.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Log file: {LOG_FILE.absolute()}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print(f"Tokens available: {len(tokens)}")
    print()
    
    log_message(logger, 'info', f"Configuration: {MAX_WORKERS} workers, {len(tokens)} tokens")
    
    # Process all files
    statistics = process_all_files(token_manager, logger)
    
    elapsed_time = time.time() - start_time
    
    print()
    print("=" * 80)
    print(f"Processing complete in {elapsed_time:.2f} seconds!")
    print("=" * 80)
    print(f"Total packages analyzed: {statistics['total_packages']}")
    print(f"Multi-repo packages found: {statistics['multirepo_count']}")
    print(f"Mono-repo packages: {statistics['monorepo_count']}")
    print(f"Skipped (forked repos): {statistics['skipped_forks']}")
    print(f"Skipped (archived repos): {statistics['skipped_archived']}")
    print(f"Errors: {statistics['error_count']}")
    print()
    print(f"Results saved in: {OUTPUT_DIR.absolute()}")
    print(f"Detailed log: {LOG_FILE.absolute()}")
    
    log_message(logger, 'info', f"Processing completed in {elapsed_time:.2f} seconds")
    log_message(logger, 'info', f"Final statistics: {statistics['total_packages']} total, "
                               f"{statistics['multirepo_count']} multi-repo, "
                               f"{statistics['monorepo_count']} mono-repo, "
                               f"{statistics['skipped_forks']} skipped (forks), "
                               f"{statistics['skipped_archived']} skipped (archived), "
                               f"{statistics['error_count']} errors")


if __name__ == "__main__":
    main()
