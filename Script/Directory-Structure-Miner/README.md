# Directory Structure Miner

This tool mines directory structures from cross-ecosystem GitHub repositories based on CSV input files.

## Features

- Processes CSV files containing cross-ecosystem package information
- Extracts directory structures from GitHub repositories using the GitHub API
- Supports multiple repository URL formats
- Handles rate limiting gracefully with **automatic token rotation**
- Outputs formatted directory trees to text files
- **NEW**: Automatically categorizes results by ecosystem count (2 ecosystems, 3 ecosystems, etc.)
- **NEW**: Generates a summary file with statistics
- **NEW**: Skips files that have already been processed
- **NEW**: Supports mining all files (unlimited depth) or limiting depth

## Setup

### 1. Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

This will:

- Create a virtual environment
- Install required dependencies (requests)
- Create the results directory

### 2. (Optional but Recommended) Set GitHub Token(s)

For higher API rate limits (5000/hour vs 60/hour), set your GitHub personal access token(s). You can provide multiple tokens separated by commas to automatically rotate through them when rate limits are reached:

```bash
export GITHUB_TOKEN='token1,token2,token3'
```

To create a GitHub token:

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scope: `public_repo` (read access to public repositories)
4. Copy the token

## Usage

### Basic Usage

Process all CSV files in the default input directory (including subdirectories):

```bash
source venv/bin/activate
python mine_directory_structure.py
```

The script will:

- Validate provided tokens and ask for confirmation
- Automatically detect and process CSV files from subdirectories (e.g., `2_ecosystems/`, `3_ecosystems/`)
- Skip files that have already been processed (output .txt files exist)
- Organize output files by ecosystem count in separate folders
- Generate a summary file with statistics

### URL-Based Mining

Mine directory structures directly from GitHub URLs without CSV files:

```bash
# Mine a single repository
python mine_directory_structure.py --url https://github.com/owner/repo --output-dir url_results

# Mine multiple repositories
python mine_directory_structure.py --url https://github.com/owner/repo1 https://github.com/owner/repo2 --output-dir url_results

# Mine repositories with custom error log directory
python mine_directory_structure.py --url https://github.com/owner/repo --output-dir my_results --error-log-dir my_results/errors
```

The URL mode:

- Accepts one or more GitHub repository URLs
- **Requires `--output-dir` to be explicitly specified** (for clarity and safety)
- Outputs results to `url_mining_results.txt` in the specified output directory
- Supports the same URL formats as CSV mode
- Does not require CSV input files
- Useful for quick analysis of specific repositories

### CSV-Based Mining (Original Mode)

```bash
# Process all CSV files
python mine_directory_structure.py

# Process specific CSV files
python mine_directory_structure.py --files Maven_Crates.csv PyPI_Crates.csv

# Use custom input/output directories
python mine_directory_structure.py --input-dir /path/to/csvs --output-dir /path/to/output

# Provide GitHub token(s) via command line
python mine_directory_structure.py --token TOKEN1 TOKEN2 TOKEN3

# Limit directory depth (default is unlimited)
python mine_directory_structure.py --max-depth 3
```

### Command-line Options

- `--url URL [URL ...]`: Mine specific GitHub repository URLs (URL mode, requires `--output-dir`)
- `--input-dir`: Directory containing CSV input files (default: `../Package-Filter/results`)
- `--output-dir`: Directory to save output files (default: `results` for CSV mode, **required** for URL mode)
- `--error-log-dir`: Directory to save error logs (default: `results/error-log`)
- `--token`: GitHub personal access token(s) (space-separated)
- `--files`: Specific CSV files to process (processes all if not specified)
- `--max-depth`: Maximum depth for directory structure (default: unlimited)
- `--help`: Show help message

## Input Format

The tool expects CSV files with columns ending in `_Repo` (e.g., `Maven_Repo`, `NPM_Repo`, `PyPI_Repo`, `Crates_Repo`).

Example CSV structure:

```csv
Maven_ID,Maven_Name,Maven_Homepage,Maven_Repo,Crates_ID,Crates_Name,Crates_Homepage,Crates_Repo
388554,tinkerforge,http://...,https://github.com/Tinkerforge/generators,53668,tinkerforge,https://...,https://github.com/Tinkerforge/generators
```

## Output Format

### Output Directory Structure

The script organizes results by ecosystem count:

```
results/
├── summary.txt                    # Summary of all mining operations
├── 2_ecosystems/                  # Results for 2-ecosystem combinations
│   ├── Crates_Go.txt
│   ├── Crates_Maven.txt
│   ├── Maven_NPM.txt
│   └── ...
├── 3_ecosystems/                  # Results for 3-ecosystem combinations
│   ├── Crates_Go_Maven.txt
│   ├── Maven_NPM_PyPI.txt
│   └── ...
├── 4_ecosystems/                  # Results for 4-ecosystem combinations
│   └── ...
├── 5_ecosystems/                  # Results for 5-ecosystem combinations
│   └── ...
└── error-log/                     # Error logs for each combination
    ├── Crates_Go_errors.log
    ├── Maven_NPM_errors.log
    └── ...
```

### Summary File

The `summary.txt` file contains:

- Overall statistics by ecosystem count
- Detailed results for each processed file
- Success rates and package counts
- List of skipped files (already processed)

Example:

```
Directory Structure Mining Summary
================================================================================

Generated: 2025-11-08 14:30:00
Total Files Processed: 45

Statistics by Ecosystem Count
--------------------------------------------------------------------------------
Count      Files      Total Pkgs      Mined Pkgs      Success Rate
--------------------------------------------------------------------------------
2          21         450             420             93.3%
3          17         180             165             91.7%
4          6          45              40              88.9%
5          1          5               5               100.0%
```

### URL Mode Output

When using `--url`, the results are saved to `url_mining_results.txt`:

```
================================================================================
Directory Structures from URLs
================================================================================
Total URLs: 2
Generated: 2025-11-03 14:30:00
================================================================================

================================================================================
Repository 1/2
================================================================================

Repository: https://github.com/Tinkerforge/generators
Owner/Repo: Tinkerforge/generators

Directory Structure:
--------------------------------------------------------------------------------

generators/
├── configs/
│   ├── bricklet_config.py
│   └── device_config.py
├── java/
│   ├── generate_java.py
│   └── templates/
└── rust/
    ├── generate_rust.py
    └── templates/
```

### CSV Mode Output

For each CSV file (e.g., `Maven_Crates.csv`), a corresponding text file is created (e.g., `Maven_Crates.txt`) with:

- Summary information (total packages, successfully mined)
- For each package:
  - Repository URL
  - Owner/Repository name
  - Ecosystems involved
  - Directory structure (tree format, max depth 3)

Example output:

```
================================================================================
Package 1/2
================================================================================

Repository: https://github.com/Tinkerforge/generators
Owner/Repo: Tinkerforge/generators
Ecosystems: Maven, Crates

Directory Structure:
--------------------------------------------------------------------------------

generators/
├── configs/
│   ├── bricklet_config.py
│   └── device_config.py
├── java/
│   ├── generate_java.py
│   └── templates/
└── rust/
    ├── generate_rust.py
    └── templates/
```

## Rate Limiting

- Without token: 60 requests/hour
- With token: 5000 requests/hour per token

The tool automatically:

- Checks rate limit status for all provided tokens
- Displays remaining requests and reset times
- **Rotates to the next available token** when one is drained
- Pauses when all tokens are exhausted
- Avoids duplicate repository requests

## Files

- `mine_directory_structure.py`: Main script
- `requirements.txt`: Python dependencies
- `setup.sh`: Setup script
- `results/`: Output directory (created automatically)

## Troubleshooting

### "Rate limit exceeded"

Wait for the rate limit to reset or use a GitHub token for higher limits.

### "Could not parse URL"

The repository URL format may not be recognized. Supported formats:

- `https://github.com/owner/repo`
- `http://github.com/owner/repo`
- `https://github.com/owner/repo.git`
- `git@github.com:owner/repo.git`

### "No CSV files found"

**In CSV mode**: Check that:

- CSV files exist in the input directory
- The input directory path is correct
- CSV files have the `.csv` extension

**In URL mode**: Use `--url` to bypass CSV file requirement.

---

## Code Explanation

### Architecture Overview

The Directory Structure Miner (REST API version) uses GitHub's REST API to fetch repository structures. The script is organized into several key components:

1. **GitHubDirectoryMiner Class**: Main class that handles API communication and tree building
2. **Error Logging System**: Comprehensive error tracking and reporting
3. **CSV Processing**: Reads cross-ecosystem package data
4. **URL Mining**: Direct repository URL processing

### 1. GitHubDirectoryMiner Class

```python
class GitHubDirectoryMiner:
    """Mines directory structures from GitHub repositories."""

    def __init__(self, github_token: Optional[str] = None, error_log_dir: Optional[str] = None):
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.headers = {}
        if self.github_token:
            self.headers['Authorization'] = f'token {self.github_token}'
        self.base_url = 'https://api.github.com'
        self.rate_limit_remaining = 60
        self.error_log_dir = error_log_dir
        self.error_logs = {}
```

**Purpose**: Manages GitHub API interactions and maintains error logs.

**Key Features**:

- Supports both authenticated and unauthenticated requests
- Tracks rate limit status
- Accumulates errors by ecosystem combination
- Uses REST API endpoint: `https://api.github.com`

### 2. URL Parsing

```python
def parse_github_url(self, url: str) -> Optional[Tuple[str, str]]:
    """Parse GitHub URL to extract owner and repo name."""
    if not url:
        return None

    url = url.rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]

    parts = url.split('github.com/')
    if len(parts) < 2:
        return None

    path_parts = parts[1].split('/')
    if len(path_parts) < 2:
        return None

    owner = path_parts[0]
    repo = path_parts[1]

    # Remove any remaining path components or query parameters
    repo = repo.split('/')[0].split('?')[0].split('#')[0]

    return (owner, repo)
```

**Process**:

1. **Validation**: Returns `None` for empty URLs
2. **Normalization**: Removes trailing slashes and `.git` suffix
3. **Extraction**: Splits on `github.com/` and extracts owner/repo
4. **Cleanup**: Removes query parameters and fragments
5. **Output**: Returns `(owner, repo)` tuple or `None`

**Examples**:

- `https://github.com/google/flatbuffers` → `('google', 'flatbuffers')`
- `https://github.com/microsoft/TypeScript.git` → `('microsoft', 'TypeScript')`

### 3. Directory Tree Fetching

```python
def get_tree(self, owner: str, repo: str, max_depth: int = 3, repo_url: str = "",
             ecosystems: List[str] = None, ecosystem_combination: str = "") -> Optional[str]:
    """Get directory tree structure from GitHub repository."""

    # Get default branch
    repo_url_api = f'{self.base_url}/repos/{owner}/{repo}'
    repo_response = requests.get(repo_url_api, headers=self.headers, timeout=10)

    if repo_response.status_code == 404:
        self.log_error(ecosystem_combination, repo_url, ecosystems,
                      "NOT_FOUND", f"Repository not found: {owner}/{repo}",
                      "Verify the repository URL is correct...")
        return None

    default_branch = repo_response.json().get('default_branch', 'master')

    # Get tree recursively
    tree_url = f'{self.base_url}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1'
    tree_response = requests.get(tree_url, headers=self.headers, timeout=10)

    tree_data = tree_response.json()
    tree_items = tree_data['tree']
    tree_dict = self._build_tree_dict(tree_items, max_depth)

    # Format as tree string
    tree_str = f"{repo}/\n"
    tree_str += self._format_tree(tree_dict, "", max_depth)

    return tree_str
```

**API Calls**:

1. **Repository Info**: `GET /repos/{owner}/{repo}` - Gets default branch name
2. **Tree Data**: `GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1` - Fetches entire repository tree

**Process**:

1. Fetch repository metadata to determine default branch
2. Fetch complete tree structure recursively
3. Build nested dictionary from flat tree list
4. Format dictionary as readable tree string
5. Apply max_depth limit to control output size

**Error Handling**:

- HTTP 404: Repository not found → logged as `NOT_FOUND`
- HTTP 403: Access denied → logged as `ACCESS_DENIED`
- Network errors: Timeout/connection issues → logged as `REQUEST_ERROR`

### 4. Tree Building

```python
def _build_tree_dict(self, items: List[Dict], max_depth: int) -> Dict:
    """Build a nested dictionary representing the tree structure."""
    tree = {}

    for item in items:
        path = item['path']
        parts = path.split('/')

        # Skip if exceeds max depth
        if len(parts) > max_depth:
            continue

        current = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Leaf node
                if item['type'] == 'tree':
                    current[part + '/'] = {}
                else:
                    current[part] = None
            else:
                # Directory node
                if part + '/' not in current:
                    current[part + '/'] = {}
                current = current[part + '/']

    return tree
```

**Purpose**: Converts flat list of paths into nested dictionary structure.

**Data Transformation**:

```python
# Input (flat list from API)
[
  {'path': 'src/main.py', 'type': 'blob'},
  {'path': 'src/utils/helper.py', 'type': 'blob'},
  {'path': 'src/utils', 'type': 'tree'}
]

# Output (nested dict)
{
  'src/': {
    'main.py': None,
    'utils/': {
      'helper.py': None
    }
  }
}
```

**Key Points**:

- Directories end with `/` in keys
- Files have `None` as value
- Respects max_depth limit
- Handles both files (`blob`) and directories (`tree`)

### 5. Error Logging

```python
def log_error(self, ecosystem_combination: str, repo_url: str, ecosystems: List[str],
              error_type: str, error_msg: str, solution: str = ""):
    """Log an error to the appropriate ecosystem combination error log."""
    if not self.error_log_dir:
        return

    timestamp = datetime.now().isoformat()

    if ecosystem_combination not in self.error_logs:
        self.error_logs[ecosystem_combination] = []

    error_entry = {
        'timestamp': timestamp,
        'repository': repo_url,
        'ecosystems': ', '.join(ecosystems),
        'error_type': error_type,
        'error_message': error_msg,
        'reason': solution if solution else error_msg,
        'solution': solution
    }

    self.error_logs[ecosystem_combination].append(error_entry)
```

**Features**:

- Organizes errors by ecosystem combination
- Records timestamp, repository, error type, and suggested solutions
- Accumulates errors in memory during processing
- Written to disk at end via `write_error_logs()`

**Error Types**:

- `NOT_FOUND`: Repository deleted/renamed
- `ACCESS_DENIED`: Private repo or rate limit
- `PARSE_ERROR`: Invalid URL format
- `API_ERROR`: GitHub API issues
- `NETWORK_ERROR`: Connection problems

### 6. URL Mining Mode

```python
def mine_urls(urls: List[str], output_dir: str, miner: GitHubDirectoryMiner):
    """Mine directory structures from provided GitHub URLs."""
    output_filename = 'url_mining_results.txt'
    output_path = os.path.join(output_dir, output_filename)

    results = []

    with tqdm(total=len(urls), desc="Mining URLs", unit="repo") as pbar:
        for url in urls:
            parsed = miner.parse_github_url(url)
            if not parsed:
                tqdm.write(f"✗ Could not parse URL: {url}")
                pbar.update(1)
                continue

            owner, repo_name = parsed
            tree = miner.get_tree(owner, repo_name, max_depth=3, repo_url=url,
                                ecosystems=[], ecosystem_combination="url_mining")

            if tree:
                results.append({
                    'repo_url': url,
                    'owner': owner,
                    'repo': repo_name,
                    'tree': tree
                })

            pbar.update(1)
```

**Purpose**: Processes URLs directly without CSV input.

**Features**:

- Accepts multiple URLs via command line
- Progress bar with repository name display
- Outputs to dedicated `url_mining_results.txt`
- Validates URLs before processing
- **Requires explicit --output-dir** for clarity

### Workflow Summary

**CSV Mode**:

1. Load CSV files containing package data
2. Build repository list per ecosystem combination
3. For each repository:
   - Parse URL to get owner/repo
   - Fetch directory tree via REST API
   - Format tree structure
   - Write to output file
4. Write error logs

**URL Mode**:

1. Parse command-line URLs
2. Validate --output-dir is specified
3. For each URL:
   - Parse to get owner/repo
   - Fetch directory tree
   - Add to results
4. Write single output file with all results
5. Write error logs

---
