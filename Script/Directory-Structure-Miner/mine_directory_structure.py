#!/usr/bin/env python3
"""
Directory Structure Miner
Mines directory structures from cross-ecosystem packages based on CSV input files.
Outputs directory trees similar to the flatbuffers_analysis.md format.
"""

import os
import csv
import requests
import base64
import time
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
DEFAULT_INPUT_DIR = DATASET_DIR / "Common-Package-Filter"
DEFAULT_OUTPUT_DIR = DATASET_DIR / "Directory-Structure-Miner"
DEFAULT_ERROR_LOG_DIR = DATASET_DIR / "Directory-Structure-Miner/error-log"

# ============================================================================


class GitHubDirectoryMiner:
    """Mines directory structures from GitHub repositories."""

    def __init__(
        self, github_tokens: List[str] = None, error_log_dir: Optional[str] = None
    ):
        """
        Initialize the miner.

        Args:
            github_tokens: List of GitHub personal access tokens
            error_log_dir: Directory to save error logs
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
        self.error_logs = {}  # Store errors by ecosystem combination

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
    ) -> Optional[str]:
        """
        Get directory tree structure from GitHub repository.

        Args:
            owner: Repository owner
            repo: Repository name
            max_depth: Maximum depth to traverse (None for unlimited)
            repo_url: Original repository URL (for error logging)
            ecosystems: List of ecosystems this repo belongs to (for error logging)
            ecosystem_combination: Ecosystem combination name (e.g., 'Maven_PyPI')
            retry_count: Number of retries attempted (for rate limit handling)

        Returns:
            Formatted directory tree string or None if failed
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

            # Build tree structure
            tree_items = tree_data["tree"]
            tree_dict = self._build_tree_dict(tree_items, max_depth)

            # Format as tree string
            tree_str = f"{repo}/\n"
            tree_str += self._format_tree(tree_dict, "", max_depth)

            return tree_str

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

    def _build_tree_dict(self, items: List[Dict], max_depth: Optional[int]) -> Dict:
        """Build a nested dictionary representing the tree structure."""
        tree = {}

        for item in items:
            path = item["path"]
            parts = path.split("/")

            # Skip if exceeds max depth
            if max_depth is not None and len(parts) > max_depth:
                continue

            current = tree
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Leaf node
                    if item["type"] == "tree":
                        current[part + "/"] = {}
                    else:
                        current[part] = None
                else:
                    # Directory node
                    if part + "/" not in current:
                        current[part + "/"] = {}
                    current = current[part + "/"]

        return tree

    def _format_tree(
        self,
        tree_dict: Dict,
        prefix: str,
        max_depth: Optional[int],
        current_depth: int = 0,
    ) -> str:
        """Format tree dictionary as a string with proper indentation."""
        if max_depth is not None and current_depth >= max_depth:
            return ""

        lines = []
        items = sorted(
            tree_dict.items(), key=lambda x: (0 if x[0].endswith("/") else 1, x[0])
        )

        for i, (name, subtree) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            lines.append(f"{prefix}{connector}{name}")

            if subtree is not None and isinstance(subtree, dict):
                lines.append(
                    self._format_tree(
                        subtree, prefix + extension, max_depth, current_depth + 1
                    )
                )

        return "\n".join(lines)


def process_csv_file(
    csv_path: str,
    output_dir: str,
    miner: GitHubDirectoryMiner,
    skip_existing: bool = False,
    max_depth: Optional[int] = None,
):
    """
    Process a CSV file and generate directory structure output.

    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save output files
        miner: GitHubDirectoryMiner instance
        skip_existing: If True, skip processing if output file already exists
        max_depth: Maximum depth to traverse (None for unlimited)
    """
    csv_filename = os.path.basename(csv_path)
    output_filename = csv_filename.replace(".csv", ".txt")
    output_path = os.path.join(output_dir, output_filename)

    # Check if output file already exists
    if skip_existing and os.path.exists(output_path):
        tqdm.write(f"⊘ Skipping {csv_filename} (output already exists)")
        return {"skipped": True, "filename": csv_filename, "count": 0}

    # Get ecosystem combination from filename (e.g., 'Maven_PyPI' from 'Maven_PyPI.csv')
    ecosystem_combination = csv_filename.replace(".csv", "")

    tqdm.write(f"\n{'=' * 80}")
    tqdm.write(f"Processing {csv_filename}")
    tqdm.write(f"{'=' * 80}")

    # Read CSV file
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        tqdm.write(f"No data in {csv_filename}")
        return

    # Extract ecosystems from CSV filename
    csv_ecosystems = csv_filename.replace(".csv", "").split("_")

    # Determine which repo columns are present based on filename
    repo_columns = [f"{eco}_Repo" for eco in csv_ecosystems]

    tqdm.write(f"Total packages: {len(rows)}")
    tqdm.write(f"Repository columns: {repo_columns}")
    tqdm.write(f"Ecosystems: {', '.join(csv_ecosystems)}\n")

    # Process each row with progress bar
    results = []

    with tqdm(total=len(rows), desc=f"Mining {csv_filename}", unit="pkg") as pbar:
        for idx, row in enumerate(rows, 1):
            # Try each repo column
            found_tree = False
            processed_repos = set()  # Track processed repos to avoid duplicates
            for repo_col in repo_columns:
                repo_url = row.get(repo_col, "").strip()

                # Skip if empty or already processed
                if not repo_url or repo_url in processed_repos:
                    continue

                processed_repos.add(repo_url)

                parsed = miner.parse_github_url(repo_url)
                if not parsed:
                    miner.log_error(
                        ecosystem_combination=ecosystem_combination,
                        repo_url=repo_url,
                        ecosystems=csv_ecosystems,
                        error_type="PARSE_ERROR",
                        error_msg=f"Could not parse GitHub URL: {repo_url}",
                        solution="Ensure the URL follows the format: https://github.com/owner/repo or git@github.com:owner/repo.git",
                    )
                    continue

                owner, repo_name = parsed
                pbar.set_postfix_str(f"{owner}/{repo_name}")

                # Determine ecosystems for this repo
                repo_ecosystems = [
                    col.replace("_Repo", "")
                    for col in repo_columns
                    if row.get(col, "").strip() == repo_url
                ]

                # Get directory tree
                tree = miner.get_tree(
                    owner,
                    repo_name,
                    max_depth=max_depth,
                    repo_url=repo_url,
                    ecosystems=repo_ecosystems,
                    ecosystem_combination=ecosystem_combination,
                )

                if tree:
                    results.append(
                        {
                            "repo_url": repo_url,
                            "owner": owner,
                            "repo": repo_name,
                            "tree": tree,
                            "ecosystems": repo_ecosystems,
                        }
                    )
                    found_tree = True
                    break  # Found valid tree, move to next row

            pbar.update(1)

            # Rate limiting check
            if idx % 10 == 0:
                miner.check_rate_limit(show_output=False)
                if miner.rate_limit_remaining < 10:
                    pbar.set_postfix_str("Rate limit low, waiting...")
                    time.sleep(60)

    # Write results to output file
    tqdm.write(f"\nWriting results to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Directory Structure Mining Results\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Source CSV: {csv_filename}\n")
        f.write(f"Total Packages: {len(rows)}\n")
        f.write(f"Successfully Mined: {len(results)}\n")
        f.write(f"{'=' * 80}\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Package {i}/{len(results)}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Repository: {result['repo_url']}\n")
            f.write(f"Owner/Repo: {result['owner']}/{result['repo']}\n")
            f.write(f"Ecosystems: {', '.join(result['ecosystems'])}\n\n")
            f.write(f"Directory Structure:\n")
            f.write(f"{'-' * 80}\n\n")
            f.write(result["tree"])
            f.write(f"\n\n")

    tqdm.write(f"✓ Results saved to {output_path}")
    tqdm.write(f"  Successfully mined: {len(results)}/{len(rows)} packages\n")

    return {
        "skipped": False,
        "filename": csv_filename,
        "total": len(rows),
        "mined": len(results),
        "ecosystems": csv_ecosystems,
        "ecosystem_count": len(csv_ecosystems),
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
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--error-log-dir",
        default=DEFAULT_ERROR_LOG_DIR,
        help=f"Directory to save error logs (default: {DEFAULT_ERROR_LOG_DIR})",
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

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()
    error_log_dir = (script_dir / args.error_log_dir).resolve()

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Directory Structure Miner")
    print(f"{'=' * 80}")
    print(f"Output directory: {output_dir}")
    print(f"Error log directory: {error_log_dir}")

    # Initialize miner
    miner = GitHubDirectoryMiner(
        github_tokens=args.token, error_log_dir=str(error_log_dir)
    )

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
        # Require output-dir to be explicitly specified in URL mode
        if args.output_dir == "results":  # default value
            print("\n" + "!" * 80)
            print("ERROR: --output-dir is required when using --url mode")
            print("!" * 80)
            print("\nPlease specify where to save the results:")
            print(
                "  python mine_directory_structure.py --url <URL> --output-dir <directory>"
            )
            print("\nExample:")
            print(
                "  python mine_directory_structure.py --url https://github.com/owner/repo --output-dir url_results"
            )
            print("=" * 80 + "\n")
            return

        mine_urls(args.url, str(output_dir), miner, max_depth=args.max_depth)

        # Write error logs
        tqdm.write(f"\n{'=' * 80}")
        tqdm.write("Writing error logs...")
        miner.write_error_logs()

        print(f"\n{'=' * 80}")
        print("URL Mining Complete!")
        print(f"{'=' * 80}")
        print(f"Results saved in: {output_dir}")
        print(f"Error logs saved in: {error_log_dir}")
        print(f"{'=' * 80}\n")
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

    # Process each CSV file and organize output by ecosystem count
    results_summary = []

    for csv_file in csv_files:
        try:
            # Determine ecosystem count from filename
            ecosystems = csv_file.stem.split("_")
            count = len(ecosystems)

            # Create subfolder for this ecosystem count
            subfolder = output_dir / f"{count}_ecosystems"
            subfolder.mkdir(parents=True, exist_ok=True)

            # Process the file
            result = process_csv_file(
                str(csv_file),
                str(subfolder),
                miner,
                skip_existing=True,
                max_depth=args.max_depth,
            )

            if result:
                results_summary.append(result)

        except Exception as e:
            tqdm.write(f"\n✗ Error processing {csv_file.name}: {e}")
            continue

    # Write error logs
    tqdm.write(f"\n{'=' * 80}")
    tqdm.write("Writing error logs...")
    miner.write_error_logs()

    # Generate summary file
    tqdm.write(f"\n{'=' * 80}")
    tqdm.write("Generating summary file...")
    summary_path = output_dir / "summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Directory Structure Mining Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Files Processed: {len(results_summary)}\n\n")

        # Group by ecosystem count
        summary_by_count = {}
        for result in results_summary:
            if result.get("skipped"):
                continue
            count = result.get("ecosystem_count", 0)
            if count not in summary_by_count:
                summary_by_count[count] = []
            summary_by_count[count].append(result)

        # Statistics by ecosystem count
        f.write("Statistics by Ecosystem Count\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Count':<10} {'Files':<10} {'Total Pkgs':<15} {'Mined Pkgs':<15} {'Success Rate':<15}\n"
        )
        f.write("-" * 80 + "\n")

        for count in sorted(summary_by_count.keys()):
            files = summary_by_count[count]
            total_packages = sum(r.get("total", 0) for r in files)
            mined_packages = sum(r.get("mined", 0) for r in files)
            success_rate = (
                f"{(mined_packages/total_packages*100):.1f}%"
                if total_packages > 0
                else "N/A"
            )

            f.write(
                f"{count:<10} {len(files):<10} {total_packages:<15} {mined_packages:<15} {success_rate:<15}\n"
            )

        f.write("\n\n")

        # Detailed results by ecosystem count
        f.write("Detailed Results\n")
        f.write("=" * 80 + "\n\n")

        for count in sorted(summary_by_count.keys()):
            f.write(f"\n{count}-Ecosystem Combinations\n")
            f.write("-" * 80 + "\n")

            files = summary_by_count[count]
            for result in sorted(files, key=lambda x: x.get("filename", "")):
                filename = result.get("filename", "Unknown")
                total = result.get("total", 0)
                mined = result.get("mined", 0)
                ecosystems = " + ".join(result.get("ecosystems", []))
                success_rate = f"{(mined/total*100):.1f}%" if total > 0 else "N/A"

                f.write(f"\nFile: {filename}\n")
                f.write(f"  Ecosystems: {ecosystems}\n")
                f.write(f"  Total Packages: {total}\n")
                f.write(f"  Mined Packages: {mined}\n")
                f.write(f"  Success Rate: {success_rate}\n")
                f.write(
                    f"  Output: {count}_ecosystems/{filename.replace('.csv', '.txt')}\n"
                )

        # Skipped files
        skipped = [r for r in results_summary if r.get("skipped")]
        if skipped:
            f.write(f"\n\nSkipped Files (Already Exist)\n")
            f.write("-" * 80 + "\n")
            for result in skipped:
                f.write(f"  • {result.get('filename', 'Unknown')}\n")

    tqdm.write(f"✓ Summary saved to {summary_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("Processing Complete!")
    print(f"{'=' * 80}")
    print(f"Results saved in: {output_dir}")
    print(f"Error logs saved in: {error_log_dir}")
    print(f"Summary saved in: {summary_path}")

    # Print statistics
    total_files = len([r for r in results_summary if not r.get("skipped")])
    skipped_files = len([r for r in results_summary if r.get("skipped")])

    print(f"\nProcessing Statistics:")
    print(f"  • Total files processed: {total_files}")
    if skipped_files > 0:
        print(f"  • Files skipped (already exist): {skipped_files}")

    if miner.error_logs:
        print(f"\nError Summary:")
        for ecosystem, errors in miner.error_logs.items():
            print(f"  • {ecosystem}: {len(errors)} errors")
    else:
        print(f"\n✓ No errors encountered")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
