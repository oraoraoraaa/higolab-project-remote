#!/usr/bin/env python3
"""
Directory Structure Miner
Mines directory structures from cross-ecosystem packages based on CSV input files.
Outputs directory trees similar to the flatbuffers_analysis.md format.
"""

import os
import sys
import csv
import json
import requests
import base64
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from datetime import datetime
from tqdm import tqdm


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Modify these paths when moving the script to another location

# Default relative paths (relative to script location)
DATASET_DIR = Path(__file__).parent / "../../Resource/Dataset/"
DEFAULT_INPUT_DIR = DATASET_DIR / "Multirepo-Common-Package-Filter"
DEFAULT_OUTPUT_FILE = DATASET_DIR / "Directory-Structure-Miner/directory_structures.json"
DEFAULT_ERROR_LOG_DIR = DATASET_DIR / "Directory-Structure-Miner/error-log"
DEFAULT_TEMP_DIR = Path(__file__).parent / "../../Temp/Directory-Structure-Miner"

# ============================================================================


def normalize_github_url(url):
    """
    Normalize GitHub repository URLs to a standard format for comparison.
    Returns None if URL is invalid, empty, or not a GitHub URL.
    This should match the normalization used in find_cross_ecosystem_packages.py
    """
    if not url or (isinstance(url, str) and url.strip() == ""):
        return None

    url = str(url).strip().lower()

    # Check if it's a GitHub URL
    if "github.com" not in url:
        return None

    # Remove common suffixes and prefixes
    url = re.sub(r"\.git$", "", url)
    url = re.sub(r"/$", "", url)
    
    # Remove git protocol prefixes
    url = url.replace('git+https://', 'https://')
    url = url.replace('git+ssh://', 'ssh://')
    url = url.replace('git://', 'https://')

    # Extract path from URL
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
            return f"github.com/{repo_path}"
    except:
        pass

    return None


class GlobalCacheIndex:
    """
    Global search index for cached directory structures.
    Builds an efficient lookup index from all temp files at startup.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the global cache index.
        
        Args:
            temp_dir: Directory containing cached directory structures
        """
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.index: Dict[str, Dict] = {}  # normalized_url -> {paths, ecosystems, source_file}
        self.loaded = False
        self.stats = {
            "total_entries": 0,
            "files_parsed": 0,
            "parse_errors": 0,
        }
    
    @staticmethod
    def _visual_tree_to_paths(tree_str: str) -> List[str]:
        """
        Convert a visual tree string to a list of paths.
        
        Args:
            tree_str: Visual tree representation with ├──, │, └── etc.
            
        Returns:
            List of file/directory paths (without the root repo name prefix)
        """
        paths = []
        path_stack = []
        root_name = None
        
        for line in tree_str.split('\n'):
            if not line.strip():
                continue
            
            clean_line = line.rstrip()
            
            # Find tree branch indicators (├── or └──)
            branch_pos = max(clean_line.rfind('├── '), clean_line.rfind('└── '))
            
            if branch_pos >= 0:
                # Regular tree item with branch indicator
                name = clean_line[branch_pos + 4:].strip()
                # Calculate depth: each level adds 4 characters of prefix (│   or    )
                # Plus the branch itself at this level
                depth = branch_pos // 4
            else:
                # Root directory line (no tree characters) - e.g., "repo-name/"
                name = clean_line.strip().rstrip('/')
                if not name:
                    continue
                # Store root name but don't add to paths (we exclude repo name from paths)
                root_name = name
                path_stack = []
                continue
            
            # Skip empty names
            if not name:
                continue
            
            # Remove trailing slash (directory indicator)
            name = name.rstrip('/')
            
            # Adjust path stack to current depth
            path_stack = path_stack[:depth]
            
            # Add current item to stack
            path_stack.append(name)
            
            # Build full path (without root repo name, matching GitHub API format)
            full_path = '/'.join(path_stack)
            if full_path:
                paths.append(full_path)
        
        return paths
    
    def build_index(self) -> None:
        """
        Build the global search index from all temp files.
        Parses all ecosystem combination files and creates a unified lookup index.
        """
        if not self.temp_dir or not self.temp_dir.exists():
            print(f"⚠ Temp directory not found: {self.temp_dir}")
            self.loaded = True
            return
        
        print(f"\n{'=' * 80}")
        print("Building Global Cache Index")
        print(f"{'=' * 80}")
        print(f"Scanning: {self.temp_dir}")
        
        # Find all ecosystem subdirectories
        ecosystem_dirs = [
            d for d in self.temp_dir.iterdir()
            if d.is_dir() and d.name.endswith("_ecosystems")
        ]
        
        if not ecosystem_dirs:
            print("⚠ No ecosystem directories found in temp dir")
            self.loaded = True
            return
        
        # Collect all txt files from all ecosystem dirs
        all_txt_files = []
        for ecosystem_dir in sorted(ecosystem_dirs):
            txt_files = list(ecosystem_dir.glob("*.txt"))
            all_txt_files.extend(txt_files)
        
        print(f"Found {len(all_txt_files)} cache files to index")
        
        # Parse each file with progress bar
        with tqdm(total=len(all_txt_files), desc="Building index", unit="file") as pbar:
            for txt_file in all_txt_files:
                pbar.set_postfix_str(txt_file.name)
                try:
                    self._parse_and_index_file(txt_file)
                    self.stats["files_parsed"] += 1
                except Exception as e:
                    tqdm.write(f"⚠ Failed to parse {txt_file.name}: {e}")
                    self.stats["parse_errors"] += 1
                pbar.update(1)
        
        self.loaded = True
        
        print(f"\n✓ Index built successfully!")
        print(f"  • Total entries: {self.stats['total_entries']:,}")
        print(f"  • Files parsed: {self.stats['files_parsed']}")
        print(f"  • Parse errors: {self.stats['parse_errors']}")
        print(f"{'=' * 80}\n")
    
    def _parse_and_index_file(self, file_path: Path) -> None:
        """
        Parse a single temp file and add entries to the index.
        
        Args:
            file_path: Path to the temp file
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by package boundaries (80 = signs followed by Package X/Y)
        # Pattern: empty line, 80 equals, newline, "Package X/Y", newline, 80 equals
        package_blocks = re.split(r'\n={80}\nPackage \d+/\d+\n={80}\n', content)
        
        # Skip the header block (first split result)
        for block in package_blocks[1:]:
            if not block.strip():
                continue
            
            # Extract repository URL
            repo_match = re.search(r'^Repository: (.+)$', block, re.MULTILINE)
            if not repo_match:
                continue
            
            repo_url = repo_match.group(1).strip()
            normalized_url = normalize_github_url(repo_url)
            
            if not normalized_url:
                continue
            
            # Extract ecosystems
            ecosystems_match = re.search(r'^Ecosystems: (.+)$', block, re.MULTILINE)
            ecosystems = []
            if ecosystems_match:
                ecosystems = [e.strip() for e in ecosystems_match.group(1).split(',')]
            
            # Extract directory structure (everything after the dashed line)
            tree_match = re.search(r'Directory Structure:\n-{80}\n\n(.+?)(?=\n\n={80}|\Z)', block, re.DOTALL)
            
            if tree_match:
                tree_str = tree_match.group(1).strip()
                
                # Convert visual tree to path list (efficient format)
                paths = self._visual_tree_to_paths(tree_str)
                
                # Only add to index if not already present (first occurrence wins)
                if normalized_url not in self.index:
                    self.index[normalized_url] = {
                        "paths": paths,
                        "ecosystems": ecosystems,
                        "source_file": file_path.name,
                    }
                    self.stats["total_entries"] += 1
    
    def lookup(self, normalized_url: str) -> Optional[List[str]]:
        """
        Look up a repository in the index.
        
        Args:
            normalized_url: Normalized GitHub URL (github.com/owner/repo)
            
        Returns:
            List of paths if found, None otherwise
        """
        if not self.loaded:
            self.build_index()
        
        entry = self.index.get(normalized_url)
        return entry["paths"] if entry else None
    
    def contains(self, normalized_url: str) -> bool:
        """
        Check if a repository is in the index.
        
        Args:
            normalized_url: Normalized GitHub URL
            
        Returns:
            True if found in index
        """
        if not self.loaded:
            self.build_index()
        return normalized_url in self.index
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return self.stats.copy()


class GitHubDirectoryMiner:
    """Mines directory structures from GitHub repositories."""

    def __init__(
        self, github_tokens: List[str] = None, error_log_dir: Optional[str] = None, temp_dir: Optional[str] = None
    ):
        """
        Initialize the miner.

        Args:
            github_tokens: List of GitHub personal access tokens
            error_log_dir: Directory to save error logs
            temp_dir: Directory to look for cached directory structures
        """
        self.tokens = github_tokens or []
        # Fallback to env var if no tokens provided
        if not self.tokens and os.environ.get("GITHUB_TOKEN"):
            env_token = os.environ.get("GITHUB_TOKEN")
            if "," in env_token:
                self.tokens = [t.strip() for t in env_token.split(",") if t.strip()]
            else:
                self.tokens = [env_token]

        self.current_token_index = 0
        self.base_url = "https://api.github.com"
        self.rate_limit_remaining = 60  # Default for unauthenticated requests
        self.error_log_dir = error_log_dir
        self.temp_dir = temp_dir
        self.error_logs = {}  # Store errors by ecosystem combination
        self.error_stats = {}  # Track error types and counts globally
        
        # Global cache index for efficient lookups
        self.global_cache = GlobalCacheIndex(temp_dir)

    @property
    def current_token(self) -> Optional[str]:
        """Get the currently active token."""
        if not self.tokens:
            return None
        return self.tokens[self.current_token_index]

    @property
    def headers(self) -> Dict[str, str]:
        """Get headers with current authentication token."""
        headers = {}
        token = self.current_token
        if token:
            headers["Authorization"] = f"token {token}"
        return headers

    def rotate_token(self) -> bool:
        """
        Switch to the next available token.
        Returns True if switched, False if no other tokens available.
        """
        if not self.tokens or len(self.tokens) <= 1:
            return False

        self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
        tqdm.write(f"⟳ Switched to token #{self.current_token_index + 1}")
        return True

    def log_error(
        self,
        ecosystem_combination: str,
        repo_url: str,
        ecosystems: List[str],
        error_type: str,
        error_msg: str,
        solution: str = "",
    ):
        """
        Log an error to the appropriate ecosystem combination error log.

        Args:
            ecosystem_combination: Ecosystem combination name (e.g., 'Maven_PyPI')
            repo_url: The repository URL that caused the error
            ecosystems: List of ecosystems this repo belongs to
            error_type: Type of error (e.g., 'PARSE_ERROR', 'API_ERROR', 'NOT_FOUND')
            error_msg: Detailed error message
            solution: Suggested solution or action
        """
        if not self.error_log_dir:
            return

        timestamp = datetime.now().isoformat()

        if ecosystem_combination not in self.error_logs:
            self.error_logs[ecosystem_combination] = []

        error_entry = {
            "timestamp": timestamp,
            "repository": repo_url,
            "ecosystems": ", ".join(ecosystems),
            "error_type": error_type,
            "error_message": error_msg,
            "reason": solution if solution else error_msg,
            "solution": solution,
        }

        self.error_logs[ecosystem_combination].append(error_entry)
        
        # Track global error statistics
        if error_type not in self.error_stats:
            self.error_stats[error_type] = 0
        self.error_stats[error_type] += 1

    def write_error_logs(self):
        """Write all accumulated error logs to files."""
        if not self.error_log_dir or not self.error_logs:
            return

        error_log_path = Path(self.error_log_dir)
        error_log_path.mkdir(parents=True, exist_ok=True)

        for ecosystem_combination, errors in self.error_logs.items():
            if not errors:
                continue

            log_file = error_log_path / f"{ecosystem_combination}_errors.log"

            with open(log_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"Error Log for {ecosystem_combination}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total Errors: {len(errors)}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                for i, error in enumerate(errors, 1):
                    f.write(f"Error #{i}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Timestamp: {error['timestamp']}\n")
                    f.write(f"Repository: {error['repository']}\n")
                    f.write(f"Ecosystems: {error['ecosystems']}\n")
                    f.write(f"Error Type: {error['error_type']}\n")
                    f.write(f"Error Message: {error['error_message']}\n")
                    f.write(f"\nPossible Reason:\n  {error['reason']}\n")
                    if error["solution"]:
                        f.write(f"\nSuggested Solution:\n  {error['solution']}\n")
                    f.write("\n" + "=" * 80 + "\n\n")

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

    def validate_tokens(self) -> None:
        """Check and display status of all provided tokens."""
        if not self.tokens:
            return

        tqdm.write("\nToken Status Check:")
        tqdm.write("-" * 65)
        tqdm.write(f"{'Token':<12} {'Status':<15} {'Remaining':<15} {'Reset Time':<15}")
        tqdm.write("-" * 65)

        for i, token in enumerate(self.tokens):
            headers = {"Authorization": f"token {token}"}
            try:
                response = requests.get(
                    f"{self.base_url}/rate_limit", headers=headers, timeout=5
                )

                if response.status_code == 200:
                    data = response.json()
                    core = data["resources"]["core"]
                    reset_time = time.strftime(
                        "%H:%M:%S", time.localtime(core["reset"])
                    )
                    status = "Valid"
                    remaining = f"{core['remaining']}/{core['limit']}"
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

            tqdm.write(f"Token #{i+1:<6} {status:<15} {remaining:<15} {reset_time:<15}")

        tqdm.write("-" * 65 + "\n")

    def check_rate_limit(self, show_output: bool = False):
        """Check and display GitHub API rate limit status. Rotates token if drained."""
        # Try up to len(tokens) times to find a usable token
        attempts = len(self.tokens) if self.tokens else 1

        for _ in range(attempts):
            try:
                response = requests.get(
                    f"{self.base_url}/rate_limit", headers=self.headers, timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    core_limit = data["resources"]["core"]
                    self.rate_limit_remaining = core_limit["remaining"]

                    if show_output:
                        tqdm.write(
                            f"Token #{self.current_token_index + 1} Rate limit: {core_limit['remaining']}/{core_limit['limit']}"
                        )
                        tqdm.write(
                            f"Resets at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(core_limit['reset']))}"
                        )

                    # If we have plenty of requests, we're good
                    if self.rate_limit_remaining > 50:
                        return

                    # If low and we have multiple tokens, rotate
                    if self.tokens and len(self.tokens) > 1:
                        if show_output:
                            tqdm.write(
                                f"Token #{self.current_token_index + 1} low ({self.rate_limit_remaining}). Rotating..."
                            )
                        self.rotate_token()
                        continue
                else:
                    if show_output:
                        tqdm.write(
                            f"Warning: API returned {response.status_code} checking rate limit"
                        )
            except Exception as e:
                if show_output:
                    tqdm.write(f"Warning: Could not check rate limit: {e}")

            # If we get here without continuing, we stop trying to rotate
            break

    def get_tree(
        self,
        owner: str,
        repo: str,
        max_depth: Optional[int] = None,
        repo_url: str = "",
        ecosystems: List[str] = None,
        ecosystem_combination: str = "",
        retry_count: int = 0,
    ) -> Optional[List[str]]:
        """
        Get directory tree structure from GitHub repository as a list of paths.

        Args:
            owner: Repository owner
            repo: Repository name
            max_depth: Maximum depth to traverse (None for unlimited)
            repo_url: Original repository URL (for error logging)
            ecosystems: List of ecosystems this repo belongs to (for error logging)
            ecosystem_combination: Ecosystem combination name (e.g., 'Maven_PyPI')
            retry_count: Number of retries attempted (for rate limit handling)

        Returns:
            List of file/directory paths or None if failed
        """
        ecosystems = ecosystems or []

        try:
            # Get default branch
            repo_url_api = f"{self.base_url}/repos/{owner}/{repo}"
            repo_response = requests.get(repo_url_api, headers=self.headers, timeout=10)

            if repo_response.status_code == 404:
                self.log_error(
                    ecosystem_combination=ecosystem_combination,
                    repo_url=repo_url or f"https://github.com/{owner}/{repo}",
                    ecosystems=ecosystems,
                    error_type="NOT_FOUND",
                    error_msg=f"Repository not found: {owner}/{repo}",
                    solution="Verify the repository URL is correct and the repository still exists. It may have been deleted, made private, or renamed.",
                )
                return None
            elif repo_response.status_code == 403:
                # Check if it's rate limit
                if (
                    "X-RateLimit-Remaining" in repo_response.headers
                    and repo_response.headers["X-RateLimit-Remaining"] == "0"
                ):
                    if self.tokens and retry_count < len(self.tokens):
                        if self.rotate_token():
                            # Retry with new token
                            return self.get_tree(
                                owner,
                                repo,
                                max_depth,
                                repo_url,
                                ecosystems,
                                ecosystem_combination,
                                retry_count + 1,
                            )

                self.log_error(
                    ecosystem_combination=ecosystem_combination,
                    repo_url=repo_url or f"https://github.com/{owner}/{repo}",
                    ecosystems=ecosystems,
                    error_type="ACCESS_DENIED",
                    error_msg=f"Access denied to repository: {owner}/{repo}",
                    solution="Repository may be private or you may have hit the rate limit. If private, ensure you have proper access credentials. If rate limited, wait for the limit to reset or use a GitHub token.",
                )
                return None
            elif repo_response.status_code != 200:
                self.log_error(
                    ecosystem_combination=ecosystem_combination,
                    repo_url=repo_url or f"https://github.com/{owner}/{repo}",
                    ecosystems=ecosystems,
                    error_type="API_ERROR",
                    error_msg=f"Error getting repo info: HTTP {repo_response.status_code}",
                    solution="Check your internet connection and GitHub API status. If the problem persists, the repository may have issues or GitHub API may be down.",
                )
                return None

            default_branch = repo_response.json().get("default_branch", "master")

            # Get tree recursively
            tree_url = f"{self.base_url}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
            tree_response = requests.get(tree_url, headers=self.headers, timeout=10)

            if tree_response.status_code != 200:
                self.log_error(
                    ecosystem_combination=ecosystem_combination,
                    repo_url=repo_url or f"https://github.com/{owner}/{repo}",
                    ecosystems=ecosystems,
                    error_type="TREE_ERROR",
                    error_msg=f"Error getting tree: HTTP {tree_response.status_code}",
                    solution="The repository tree may be too large or the default branch may be empty. Try cloning the repository manually to inspect its structure.",
                )
                return None

            tree_data = tree_response.json()

            if "tree" not in tree_data:
                self.log_error(
                    ecosystem_combination=ecosystem_combination,
                    repo_url=repo_url or f"https://github.com/{owner}/{repo}",
                    ecosystems=ecosystems,
                    error_type="TREE_EMPTY",
                    error_msg="Repository tree is empty or malformed",
                    solution="The repository may be empty or newly created. Check if it contains any files or commits.",
                )
                return None

            # Build tree structure and return as path list (efficient format)
            tree_items = tree_data["tree"]
            paths = [item["path"] for item in tree_items]
            return paths

        except requests.exceptions.Timeout as e:
            self.log_error(
                ecosystem_combination=ecosystem_combination,
                repo_url=repo_url or f"https://github.com/{owner}/{repo}",
                ecosystems=ecosystems,
                error_type="TIMEOUT",
                error_msg=f"Request timeout: {str(e)}",
                solution="The GitHub API request timed out. This may be due to network issues or a very large repository. Try again later or increase the timeout value.",
            )
            return None
        except requests.exceptions.RequestException as e:
            self.log_error(
                ecosystem_combination=ecosystem_combination,
                repo_url=repo_url or f"https://github.com/{owner}/{repo}",
                ecosystems=ecosystems,
                error_type="REQUEST_ERROR",
                error_msg=f"Request error: {str(e)}",
                solution="Network error occurred while accessing GitHub API. Check your internet connection and try again.",
            )
            return None
        except Exception as e:
            self.log_error(
                ecosystem_combination=ecosystem_combination,
                repo_url=repo_url or f"https://github.com/{owner}/{repo}",
                ecosystems=ecosystems,
                error_type="UNKNOWN_ERROR",
                error_msg=f"Unexpected error: {str(e)}",
                solution="An unexpected error occurred. Please check the error details and report if this is a bug.",
            )
            return None

    def lookup_cache(self, normalized_url: str) -> Optional[List[str]]:
        """
        Look up a repository in the global cache index.
        
        Args:
            normalized_url: Normalized GitHub URL (github.com/owner/repo)
            
        Returns:
            List of paths if found in cache, None otherwise
        """
        return self.global_cache.lookup(normalized_url)
    
    def is_cached(self, normalized_url: str) -> bool:
        """
        Check if a repository is in the global cache.
        
        Args:
            normalized_url: Normalized GitHub URL
            
        Returns:
            True if found in cache
        """
        return self.global_cache.contains(normalized_url)


def save_checkpoint(output_file: Path, all_results: List[Dict], all_packages: List[Dict], total_processed: int, total_mined: int, total_cache_hits: int):
    """
    Save current progress to JSON file as a checkpoint.
    
    Args:
        output_file: Path to output JSON file
        all_results: List of combination results
        all_packages: List of all package data
        total_processed: Total packages processed so far
        total_mined: Total packages successfully mined
        total_cache_hits: Total cache hits
    """
    json_output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "total_combinations": len(all_results),
            "total_packages_processed": total_processed,
            "total_packages_mined": total_mined,
            "total_cache_hits": total_cache_hits,
            "cache_hit_rate": f"{total_cache_hits/total_processed*100:.2f}%" if total_processed > 0 else "0%",
            "status": "in_progress",
        },
        "combinations": all_results,
        "packages": all_packages,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)


def load_checkpoint(output_file: Path) -> Optional[Dict]:
    """
    Load existing checkpoint from output file.
    
    Args:
        output_file: Path to output JSON file
        
    Returns:
        Checkpoint data if exists and valid, None otherwise
    """
    if not output_file.exists():
        return None
    
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate checkpoint structure
        if "metadata" not in data or "packages" not in data:
            return None
        
        return data
    except (json.JSONDecodeError, IOError) as e:
        tqdm.write(f"⚠ Failed to load checkpoint: {e}")
        return None


def get_processed_repos_from_checkpoint(checkpoint_data: Optional[Dict]) -> set:
    """
    Extract set of already processed repository URLs from checkpoint.
    
    Args:
        checkpoint_data: Loaded checkpoint data
        
    Returns:
        Set of normalized repository URLs that have been processed
    """
    if not checkpoint_data:
        return set()
    
    processed = set()
    for package in checkpoint_data.get("packages", []):
        repo = package.get("repository", "")
        if repo:
            processed.add(repo)
    
    return processed


def process_csv_file(
    csv_path: str,
    miner: GitHubDirectoryMiner,
    output_file: Path,
    all_results: List[Dict],
    all_packages: List[Dict],
    total_processed: int,
    total_mined: int,
    total_cache_hits: int,
    processed_repos: set,
    max_depth: Optional[int] = None,
) -> Dict:
    """
    Process a CSV file and return structured directory data.
    Saves checkpoints every 10 packages. Skips already processed repositories.

    Args:
        csv_path: Path to input CSV file
        miner: GitHubDirectoryMiner instance
        output_file: Path to output JSON file for checkpoints
        all_results: List of all combination results (modified in place)
        all_packages: List of all package data (modified in place)
        total_processed: Running total of packages processed
        total_mined: Running total of packages mined
        total_cache_hits: Running total of cache hits
        processed_repos: Set of already processed repository URLs (for checkpoint resume)
        max_depth: Maximum depth to traverse (None for unlimited)
        
    Returns:
        Dictionary containing processing results and package data
    """
    csv_filename = os.path.basename(csv_path)
    
    # Get ecosystem combination from filename (e.g., 'Maven_PyPI' from 'Maven_PyPI.csv')
    ecosystem_combination = csv_filename.replace(".csv", "")
    
    # Extract claimed ecosystems from filename
    claimed_ecosystems = ecosystem_combination.split("_")

    tqdm.write(f"\n{'=' * 80}")
    tqdm.write(f"Processing {csv_filename}")
    tqdm.write(f"{'=' * 80}")

    # Read CSV file
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        tqdm.write(f"No data in {csv_filename}")
        return None

    # Count how many need to be processed (excluding already processed and those in global cache)
    repos_to_process = []
    repos_from_checkpoint = []
    repos_from_cache = []
    
    for row in rows:
        normalized_url = row.get("Normalized_URL", "").strip()
        if not normalized_url:
            continue
        
        if normalized_url in processed_repos:
            repos_from_checkpoint.append(normalized_url)
        elif miner.is_cached(normalized_url):
            repos_from_cache.append(normalized_url)
        else:
            repos_to_process.append(normalized_url)

    tqdm.write(f"Total packages: {len(rows)}")
    tqdm.write(f"  • Already processed (checkpoint): {len(repos_from_checkpoint)}")
    tqdm.write(f"  • Available in cache: {len(repos_from_cache)}")
    tqdm.write(f"  • Need API mining: {len(repos_to_process)}")
    tqdm.write(f"Claimed Ecosystems: {', '.join(claimed_ecosystems)}\n")

    # Process each row with progress bar
    packages_data = []
    local_processed = 0
    local_mined = 0
    local_cache_hits = 0
    skipped_checkpoint = 0

    with tqdm(total=len(rows), desc=f"Mining {csv_filename}", unit="pkg") as pbar:
        for idx, row in enumerate(rows, 1):
            # Only use Normalized_URL column - no fallback
            normalized_url = row.get("Normalized_URL", "").strip()
            
            if not normalized_url:
                miner.log_error(
                    ecosystem_combination=ecosystem_combination,
                    repo_url="Missing",
                    ecosystems=claimed_ecosystems,
                    error_type="MISSING_NORMALIZED_URL",
                    error_msg="Normalized_URL column is missing or empty",
                    solution="Ensure the CSV file was generated with the updated find_cross_ecosystem_packages.py that includes Normalized_URL column.",
                )
                pbar.update(1)
                continue
            
            # Skip already processed repositories (from checkpoint)
            if normalized_url in processed_repos:
                skipped_checkpoint += 1
                pbar.set_postfix_str(f"[SKIP] {normalized_url.split('/')[-1]}")
                pbar.update(1)
                continue
            
            # Convert normalized URL (github.com/owner/repo) to full URL
            full_url = f"https://{normalized_url}"
            parsed = miner.parse_github_url(full_url)
            
            if not parsed:
                miner.log_error(
                    ecosystem_combination=ecosystem_combination,
                    repo_url=full_url,
                    ecosystems=claimed_ecosystems,
                    error_type="PARSE_ERROR",
                    error_msg=f"Could not parse normalized URL: {normalized_url}",
                    solution="Check if the normalized URL format is correct (should be github.com/owner/repo).",
                )
                pbar.update(1)
                continue
            
            owner, repo_name = parsed
            pbar.set_postfix_str(f"{owner}/{repo_name}")
            
            # Check if already cached in global cache index (from Temp directory)
            paths = None
            cached_from_index = False
            
            cached_paths = miner.lookup_cache(normalized_url)
            if cached_paths:
                paths = cached_paths
                cached_from_index = True
                local_cache_hits += 1
                pbar.set_postfix_str(f"{owner}/{repo_name} [CACHED]")
            else:
                # Get directory tree from GitHub API
                paths = miner.get_tree(
                    owner,
                    repo_name,
                    max_depth=max_depth,
                    repo_url=full_url,
                    ecosystems=claimed_ecosystems,
                    ecosystem_combination=ecosystem_combination,
                )
            
            if paths:
                package_entry = {
                    "repository": normalized_url,
                    "claimed_ecosystems": claimed_ecosystems,
                    "directory_structure": paths,
                    "cached": cached_from_index,
                }
                packages_data.append(package_entry)
                all_packages.append(package_entry)
                local_mined += 1
                
                # Add to processed repos to avoid re-processing
                processed_repos.add(normalized_url)
            else:
                miner.log_error(
                    ecosystem_combination=ecosystem_combination,
                    repo_url=full_url,
                    ecosystems=claimed_ecosystems,
                    error_type="TREE_FETCH_FAILED",
                    error_msg=f"Failed to fetch directory tree for {owner}/{repo_name}",
                    solution="Repository may not exist, be private, or API may have failed. Check error logs for details.",
                )

            local_processed += 1
            pbar.update(1)

            # Save checkpoint every 10 packages (only for non-skipped packages)
            if local_processed % 10 == 0 and local_mined > 0:
                # Update running totals
                current_total_processed = total_processed + local_processed
                current_total_mined = total_mined + local_mined
                current_total_cache_hits = total_cache_hits + local_cache_hits
                
                # Update this combination's stats in all_results
                current_result = {
                    "combination": ecosystem_combination,
                    "claimed_ecosystems": claimed_ecosystems,
                    "total_packages": len(rows),
                    "mined_packages": local_mined,
                    "cache_hits": local_cache_hits,
                    "skipped_checkpoint": skipped_checkpoint,
                }
                
                # Update or add to all_results
                existing_idx = None
                for i, r in enumerate(all_results):
                    if r.get("combination") == ecosystem_combination:
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    all_results[existing_idx] = current_result
                else:
                    all_results.append(current_result)
                
                save_checkpoint(output_file, all_results, all_packages, 
                               current_total_processed, current_total_mined, current_total_cache_hits)
                tqdm.write(f"  💾 Checkpoint saved: {local_processed}/{len(rows)} packages from {csv_filename}")
            
            # Rate limiting check (only for API calls, not cached entries)
            if idx % 10 == 0 and not cached_from_index:
                miner.check_rate_limit(show_output=False)
                if miner.rate_limit_remaining < 10:
                    pbar.set_postfix_str("Rate limit low, waiting...")
                    time.sleep(60)

    # Report results
    tqdm.write(f"\n✓ Completed {csv_filename}:")
    tqdm.write(f"  • Processed: {local_processed}")
    tqdm.write(f"  • Mined: {local_mined}")
    tqdm.write(f"  • Cache hits: {local_cache_hits}")
    tqdm.write(f"  • Skipped (checkpoint): {skipped_checkpoint}")
    if local_cache_hits > 0 and local_processed > 0:
        tqdm.write(f"  • Cache hit rate: {local_cache_hits/local_processed*100:.1f}%")

    return {
        "combination": ecosystem_combination,
        "claimed_ecosystems": claimed_ecosystems,
        "total_packages": len(rows),
        "mined_packages": local_mined,
        "cache_hits": local_cache_hits,
        "skipped_checkpoint": skipped_checkpoint,
        "local_processed": local_processed,
        "local_mined": local_mined,
        "local_cache_hits": local_cache_hits,
    }


def mine_urls(
    urls: List[str],
    output_dir: str,
    miner: GitHubDirectoryMiner,
    max_depth: Optional[int] = None,
):
    """
    Mine directory structures from provided GitHub URLs.

    Args:
        urls: List of GitHub repository URLs
        output_dir: Directory to save output files
        miner: GitHubDirectoryMiner instance
        max_depth: Maximum depth to traverse (None for unlimited)
    """
    output_filename = "url_mining_results.txt"
    output_path = os.path.join(output_dir, output_filename)

    tqdm.write(f"\n{'=' * 80}")
    tqdm.write(f"Mining Directory Structures from URLs")
    tqdm.write(f"{'=' * 80}")
    tqdm.write(f"Total URLs: {len(urls)}\n")

    # Process each URL with progress bar
    results = []

    with tqdm(total=len(urls), desc="Mining URLs", unit="repo") as pbar:
        for url in urls:
            parsed = miner.parse_github_url(url)
            if not parsed:
                tqdm.write(f"✗ Could not parse URL: {url}")
                pbar.update(1)
                continue

            owner, repo_name = parsed
            pbar.set_postfix_str(f"{owner}/{repo_name}")

            # Get directory tree
            tree = miner.get_tree(
                owner,
                repo_name,
                max_depth=max_depth,
                repo_url=url,
                ecosystems=[],
                ecosystem_combination="url_mining",
            )

            if tree:
                results.append(
                    {"repo_url": url, "owner": owner, "repo": repo_name, "tree": tree}
                )

            pbar.update(1)

            # Rate limiting check
            if len(results) % 10 == 0:
                miner.check_rate_limit(show_output=False)
                if miner.rate_limit_remaining < 10:
                    pbar.set_postfix_str("Rate limit low, waiting...")
                    time.sleep(60)

    # Write results to output file
    tqdm.write(f"\nWriting results to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Directory Structure Mining Results (URL Mode)\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Total URLs Provided: {len(urls)}\n")
        f.write(f"Successfully Mined: {len(results)}\n")
        f.write(f"{'=' * 80}\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Repository {i}/{len(results)}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Repository: {result['repo_url']}\n")
            f.write(f"Owner/Repo: {result['owner']}/{result['repo']}\n\n")
            f.write(f"Directory Structure:\n")
            f.write(f"{'-' * 80}\n\n")
            f.write(result["tree"])
            f.write(f"\n\n")

    tqdm.write(f"✓ Results saved to {output_path}")
    tqdm.write(f"  Successfully mined: {len(results)}/{len(urls)} repositories\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mine directory structures from cross-ecosystem packages or GitHub URLs"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing CSV input files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--error-log-dir",
        default=DEFAULT_ERROR_LOG_DIR,
        help=f"Directory to save error logs (default: {DEFAULT_ERROR_LOG_DIR})",
    )
    parser.add_argument(
        "--temp-dir",
        default=DEFAULT_TEMP_DIR,
        help=f"Directory containing cached directory structures (default: {DEFAULT_TEMP_DIR})",
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
        "--url",
        nargs="+",
        metavar="URL",
        help="Mine specific GitHub repository URLs instead of CSV files",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum depth for directory structure (default: unlimited)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache lookup, always fetch from API",
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    output_file = (script_dir / args.output_file).resolve()
    error_log_dir = (script_dir / args.error_log_dir).resolve()
    temp_dir = (script_dir / args.temp_dir).resolve()

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Directory Structure Miner")
    print(f"{'=' * 80}")
    print(f"Output file: {output_file}")
    print(f"Error log directory: {error_log_dir}")
    print(f"Cache directory: {temp_dir}")
    if args.no_resume:
        print(f"Resume mode: DISABLED (--no-resume)")
    if args.no_cache:
        print(f"Cache mode: DISABLED (--no-cache)")
    
    # Initialize miner (use None for temp_dir if cache is disabled)
    miner = GitHubDirectoryMiner(
        github_tokens=args.token, 
        error_log_dir=str(error_log_dir), 
        temp_dir=None if args.no_cache else str(temp_dir)
    )

    # Prompt for tokens if not provided
    if not miner.tokens:
        print("\n" + "=" * 80)
        print("GitHub Personal Access Token Required")
        print("=" * 80)
        print("To use the GitHub API, you need to provide at least one Personal Access Token.")
        print("\nYou can:")
        print("  1. Set the GITHUB_TOKEN environment variable")
        print("  2. Use the --token option when running the script")
        print("  3. Enter tokens interactively now\n")
        
        response = input("Would you like to enter tokens now? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            tokens = []
            print("\nEnter tokens (one per line, press Enter with empty line when done):")
            while True:
                token = input(f"Token {len(tokens) + 1}: ").strip()
                if not token:
                    break
                tokens.append(token)
            
            if tokens:
                miner.tokens = tokens
                miner.current_token_index = 0
                print(f"\n✓ Added {len(tokens)} token(s)")
            else:
                print("\n⚠ No tokens provided. Exiting...")
                sys.exit(1)
        else:
            print("\n⚠ Cannot proceed without authentication. Exiting...")
            sys.exit(1)

    if miner.tokens:
        miner.validate_tokens()
        try:
            response = (
                input("Do you want to proceed with these tokens? (yes/no): ")
                .strip()
                .lower()
            )
            if response not in ["yes", "y"]:
                print("\n✗ Operation cancelled by user.")
                return
            # Prime the rate limit check for the first token
            miner.check_rate_limit(show_output=False)
        except (KeyboardInterrupt, EOFError):
            print("\n\n✗ Operation cancelled by user.")
            return
    else:
        print(f"\nChecking GitHub API rate limit...")
        miner.check_rate_limit(show_output=True)

    if not miner.tokens:
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
                print(
                    "\n✗ Operation cancelled. Please provide a GitHub token and try again."
                )
                return
            print("\n✓ Continuing without token...")
        except (KeyboardInterrupt, EOFError):
            print("\n\n✗ Operation cancelled by user.")
            return

    # Validate tokens before processing
    miner.validate_tokens()

    # Check if URL mode is enabled
    if args.url:
        print("\n" + "!" * 80)
        print("ERROR: URL mode is not yet supported with JSON output")
        print("!" * 80)
        print("\nPlease use CSV mode instead.")
        print("=" * 80 + "\n")
        return

    # CSV mode (original functionality)
    input_dir = (script_dir / args.input_dir).resolve()
    print(f"Input directory: {input_dir}")

    # Get CSV files to process from all subdirectories
    csv_files = []
    if args.files:
        # If specific files provided, look for them in the input directory or its subdirectories
        for f in args.files:
            file_path = input_dir / f
            if file_path.is_file():
                csv_files.append(file_path)
            else:
                # Try to find in subdirectories
                found = list(input_dir.glob(f"**/{f}"))
                if found:
                    csv_files.extend(found)
    else:
        # Get all CSV files from subdirectories (e.g., 2_ecosystems/, 3_ecosystems/, etc.)
        ecosystem_dirs = [
            d
            for d in input_dir.iterdir()
            if d.is_dir() and d.name.endswith("_ecosystems")
        ]

        if ecosystem_dirs:
            # New structure with ecosystem folders
            for ecosystem_dir in sorted(ecosystem_dirs):
                csv_files.extend(list(ecosystem_dir.glob("*.csv")))
        else:
            # Old structure with CSV files directly in input_dir
            csv_files = list(input_dir.glob("*.csv"))

    csv_files = [f for f in csv_files if f.is_file()]

    if not csv_files:
        print(f"\n✗ No CSV files found in {input_dir}")
        return

    print(f"\nFound {len(csv_files)} CSV file(s) to process:")

    # Group by ecosystem count for display
    files_by_count = {}
    for f in csv_files:
        # Determine ecosystem count from filename
        ecosystems = f.stem.split("_")
        count = len(ecosystems)
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

    # Parse all CSV files to count total packages
    print(f"\n{'=' * 80}")
    print("Parsing input files to count packages...")
    print(f"{'=' * 80}")
    
    total_packages = 0
    files_package_counts = {}
    
    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Count valid packages (those with Normalized_URL)
            valid_count = 0
            for row in rows:
                normalized_url = row.get("Normalized_URL", "").strip()
                if normalized_url:
                    # Validate format
                    parts = normalized_url.replace('github.com/', '').split('/')
                    if len(parts) == 2 and parts[0] and parts[1]:
                        valid_count += 1
            
            files_package_counts[csv_file] = valid_count
            total_packages += valid_count
    
    print(f"\nTotal packages to process: {total_packages:,}")
    print(f"Total CSV files: {len(csv_files)}")
    
    # Group counts by ecosystem count
    counts_by_ecosystem = {}
    for csv_file, count in files_package_counts.items():
        ecosystem_count = len(csv_file.stem.split("_"))
        if ecosystem_count not in counts_by_ecosystem:
            counts_by_ecosystem[ecosystem_count] = {'files': 0, 'packages': 0}
        counts_by_ecosystem[ecosystem_count]['files'] += 1
        counts_by_ecosystem[ecosystem_count]['packages'] += count
    
    print(f"\nPackages by ecosystem count:")
    for count in sorted(counts_by_ecosystem.keys()):
        stats = counts_by_ecosystem[count]
        print(f"  {count}-ecosystems: {stats['packages']:,} packages from {stats['files']} files")
    
    # Build global cache index from temp directory (unless --no-cache is set)
    if not args.no_cache:
        print(f"\n{'=' * 80}")
        print("Building Global Cache Index...")
        print(f"{'=' * 80}")
        miner.global_cache.build_index()
    else:
        print(f"\n{'=' * 80}")
        print("Cache disabled (--no-cache). Skipping cache index build.")
        print(f"{'=' * 80}")
    
    # Check for existing checkpoint and load resume state (unless --no-resume is set)
    processed_repos = set()
    
    if not args.no_resume:
        checkpoint_data = load_checkpoint(output_file)
        processed_repos = get_processed_repos_from_checkpoint(checkpoint_data)
        
        if checkpoint_data:
            checkpoint_status = checkpoint_data.get("metadata", {}).get("status", "unknown")
            checkpoint_processed = checkpoint_data.get("metadata", {}).get("total_packages_processed", 0)
            checkpoint_mined = checkpoint_data.get("metadata", {}).get("total_packages_mined", 0)
            
            print(f"\n{'=' * 80}")
            print("Checkpoint Found!")
            print(f"{'=' * 80}")
            print(f"  • Status: {checkpoint_status}")
            print(f"  • Already processed: {checkpoint_processed:,} packages")
            print(f"  • Already mined: {checkpoint_mined:,} packages")
            print(f"  • Unique repos in checkpoint: {len(processed_repos):,}")
            
            if checkpoint_status == "completed":
                print(f"\n⚠ Previous run was marked as completed.")
                try:
                    response = input("Do you want to re-run anyway? (y/n): ").strip().lower()
                    if response not in ['y', 'yes']:
                        print("\n✓ Exiting. Previous results are already complete.")
                        sys.exit(0)
                    # Clear checkpoint to start fresh
                    processed_repos = set()
                    print("\n✓ Starting fresh run (ignoring checkpoint)...")
                except (KeyboardInterrupt, EOFError):
                    print("\n\n⚠ Cancelled by user. Exiting...")
                    sys.exit(0)
            else:
                try:
                    response = input("\nResume from checkpoint? (y/n): ").strip().lower()
                    if response in ['y', 'yes']:
                        print(f"\n✓ Resuming from checkpoint. Will skip {len(processed_repos):,} already processed repos.")
                    else:
                        processed_repos = set()
                        print("\n✓ Starting fresh run (ignoring checkpoint)...")
                except (KeyboardInterrupt, EOFError):
                    print("\n\n⚠ Cancelled by user. Exiting...")
                    sys.exit(0)
    else:
        print(f"\n{'=' * 80}")
        print("Resume disabled (--no-resume). Starting fresh.")
        print(f"{'=' * 80}")
    
    # Calculate how many can be served from cache vs need API
    cache_available = 0
    need_api = 0
    already_processed = 0
    
    for csv_file, count in files_package_counts.items():
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized_url = row.get("Normalized_URL", "").strip()
                if normalized_url:
                    if normalized_url in processed_repos:
                        already_processed += 1
                    elif miner.is_cached(normalized_url):
                        cache_available += 1
                    else:
                        need_api += 1
    
    print(f"\n{'=' * 80}")
    print("Processing Plan")
    print(f"{'=' * 80}")
    print(f"  • Total packages: {total_packages:,}")
    print(f"  • Already processed (checkpoint): {already_processed:,}")
    print(f"  • Available in cache: {cache_available:,}")
    print(f"  • Need API mining: {need_api:,}")
    
    if need_api > 0:
        estimated_api_time = need_api * 2  # ~2 seconds per API call
        print(f"  • Estimated API time: ~{estimated_api_time // 60} min {estimated_api_time % 60} sec")
    
    # Ask for confirmation
    print(f"\n{'=' * 80}")
    try:
        response = input("\nProceed with mining? (y/n): ").strip().lower()
        
        if response not in ['y', 'yes']:
            print("\n⚠ Mining cancelled by user. Exiting...")
            sys.exit(0)
    except (KeyboardInterrupt, EOFError):
        print("\n\n⚠ Mining cancelled by user. Exiting...")
        sys.exit(0)
    
    print(f"\n{'=' * 80}")
    print("Starting mining process...")
    print(f"{'=' * 80}")

    # Process each CSV file and collect all results
    all_results = []
    all_packages = []
    total_processed = 0
    total_mined = 0
    total_cache_hits = 0

    for csv_file in csv_files:
        try:
            result = process_csv_file(
                str(csv_file),
                miner,
                output_file,
                all_results,
                all_packages,
                total_processed,
                total_mined,
                total_cache_hits,
                processed_repos,
                max_depth=args.max_depth,
            )
            
            if result:
                # Update or replace result for this combination
                existing_idx = None
                for i, r in enumerate(all_results):
                    if r.get("combination") == result["combination"]:
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    all_results[existing_idx] = result
                else:
                    all_results.append(result)
                
                # Update totals
                total_processed += result["local_processed"]
                total_mined += result["local_mined"]
                total_cache_hits += result["local_cache_hits"]

        except Exception as e:
            tqdm.write(f"\n✗ Error processing {csv_file.name}: {e}")
            continue

    # Write error logs
    tqdm.write(f"\n{'=' * 80}")
    tqdm.write("Writing error logs...")
    miner.write_error_logs()

    # Write final JSON output
    tqdm.write(f"\n{'=' * 80}")
    tqdm.write(f"Writing final results to {output_file}...")
    
    json_output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "total_combinations": len(all_results),
            "total_packages_processed": total_processed,
            "total_packages_mined": total_mined,
            "total_cache_hits": total_cache_hits,
            "cache_hit_rate": f"{total_cache_hits/total_processed*100:.2f}%" if total_processed > 0 else "0%",
            "status": "completed",
        },
        "combinations": all_results,
        "packages": all_packages,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    tqdm.write(f"✓ Saved {len(all_packages)} packages to {output_file} (FINAL)")

    # Summary
    print(f"\n{'=' * 80}")
    print("Processing Complete!")
    print(f"{'=' * 80}")
    print(f"Results saved in: {output_file}")
    print(f"Error logs saved in: {error_log_dir}")

    # Print statistics
    print(f"\nProcessing Statistics:")
    print(f"  • Total combinations processed: {len(all_results)}")
    print(f"  • Total packages processed: {total_processed:,}")
    print(f"  • Successfully mined: {total_mined:,} ({total_mined/total_processed*100:.1f}%)" if total_processed > 0 else "  • Successfully mined: 0")
    print(f"  • Cache hits: {total_cache_hits:,} ({total_cache_hits/total_processed*100:.1f}%)" if total_processed > 0 else "  • Cache hits: 0")

    if miner.error_logs:
        print(f"\nError Summary:")
        for ecosystem, errors in miner.error_logs.items():
            print(f"  • {ecosystem}: {len(errors)} errors")
    else:
        print(f"\n✓ No errors encountered")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
