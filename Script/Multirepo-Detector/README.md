# Multi-Repo Detector

This script detects packages that follow a multi-repository pattern (separate repos for each language/ecosystem) vs. mono-repository pattern (single repo with multiple languages).

## Overview

The script analyzes directory structure mining results and uses the GitHub API to detect if a package follows the multi-repo pattern where client libraries are in separate repositories like:

- `package-python`
- `package-rust`
- `package-go`
- `package-js`
- etc.

## Detection Logic

For each package, the script:

1. **Extracts owner/repo** from the GitHub URL
2. **Fetches all repositories** from that GitHub user/organization
3. **Checks for multi-repo patterns** using common separators (-, \_, .)
4. **Identifies ecosystem suffixes** like: python, rust, go, js, java, php, ruby, net, cpp, etc.
5. **Detects multi-repo** if 2+ repos match the pattern `<base><separator><ecosystem>`

## Example

For `https://github.com/logbull/logbull`:

- Checks all `logbull` organization repos
- Finds: `logbull-python`, `logbull-rust`, `logbull-go`, `logbull-js`, etc.
- Detects as **multi-repo** with separator `-`

## Requirements

```bash
pip install requests
```

## Usage

### Basic Usage

```bash
python detect_multirepo.py
```

### With GitHub Token (Recommended)

For higher rate limits (5,000 requests/hour instead of 60):

```bash
export GITHUB_TOKEN="your_github_token"
python detect_multirepo.py
```

### With Multiple Tokens (Best Performance)

For maximum speed, use multiple GitHub tokens separated by commas:

```bash
export GITHUB_TOKEN="token1,token2,token3"
python detect_multirepo.py
```

Or enter them when prompted (comma or space-separated):

```
Enter GitHub token(s): token1,token2,token3
```

**Benefits of multiple tokens:**

- Each token provides 5,000 requests/hour
- Tokens are used in round-robin fashion for parallel requests
- 3 tokens = 15,000 effective requests/hour
- Significantly reduces processing time for large datasets

To create GitHub tokens:

1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select scope: `public_repo` (read-only access to public repositories)
4. Repeat to create multiple tokens

## Input

Reads from: `../Directory-Structure-Miner/results/`

Expected structure:

```
Directory-Structure-Miner/results/
├── 2_ecosystems/
│   ├── Crates_Go.txt
│   ├── Go_NPM.txt
│   └── ...
├── 3_ecosystems/
│   ├── Crates_Go_NPM.txt
│   └── ...
└── ...
```

## Output

### 1. Multi-Repo Packages CSV

`results/multirepo_packages.csv` contains all detected multi-repo packages:

| name    | root_url                           | ecosystems   | separator | found_repos                       |
| ------- | ---------------------------------- | ------------ | --------- | --------------------------------- |
| logbull | https://github.com/logbull/logbull | Crates, PyPI | -         | logbull-python, logbull-rust, ... |

### 2. Filtered Directory Structure Files

Creates the same directory structure as input, but with multi-repo packages removed:

```
results/
├── 2_ecosystems/
│   ├── Crates_Go.txt (multi-repo packages removed)
│   ├── Go_NPM.txt (multi-repo packages removed)
│   └── ...
├── 3_ecosystems/
│   └── ...
└── multirepo_packages.csv
```

### 3. Summary Statistics

`results/summary.txt` contains:

- Total packages analyzed
- Multi-repo vs mono-repo counts
- Per-file statistics
- Processing details

## Features

- **Parallel Processing**: Processes multiple packages concurrently using ThreadPoolExecutor (10 workers by default)
- **Response Caching**: Caches GitHub API responses to avoid redundant calls
- **Token Round-Robin**: Distributes API calls across multiple tokens for maximum throughput
- **Thread-Safe Operations**: Uses locks for safe concurrent access to shared resources
- **Automatic rate limiting**: Respects GitHub API limits
- **Multiple separator detection**: Checks -, \_, and . separators
- **Comprehensive ecosystem matching**: Detects 15+ common ecosystems
- **Error handling**: Continues processing on failures
- **Progress reporting**: Real-time status updates
- **Original format preservation**: Output files maintain input format

## Performance

The script has been optimized for speed:

- **Parallel API calls**: Up to 10 concurrent requests (configurable via `MAX_WORKERS`)
- **Intelligent caching**: Reuses API responses for packages from the same owner
- **Token distribution**: Multiple tokens can be used in parallel for higher effective rate limits
- **Expected speedup**: 5-10x faster than sequential processing

### Performance Tips

1. **Use multiple GitHub tokens**: Each token gets 5,000 requests/hour. Use 3-5 tokens for best performance.
2. **Adjust worker count**: Modify `MAX_WORKERS` in the script if needed (default: 10)
3. **Fast internet connection**: Network latency affects parallel performance

## GitHub API Rate Limits

| Type            | Limit               | Reset      |
| --------------- | ------------------- | ---------- |
| Unauthenticated | 60 requests/hour    | Every hour |
| Authenticated   | 5,000 requests/hour | Every hour |

The script automatically:

- Monitors rate limit usage
- Waits when approaching limits
- Uses pagination for users with many repos

## Implementation Details

### Separator Detection

GitHub repository names can only contain: `[a-zA-Z0-9._-]`

The script checks all three valid separators:

- `-` (most common): `package-python`
- `_` (underscore): `package_python`
- `.` (dot): `package.python`

### Base Name Extraction

The script tries multiple strategies to find the base name:

1. Use full repo name as base
2. Remove ecosystem suffix from end: `logbull-python` → `logbull`
3. Remove ecosystem prefix from start: `python-logbull` → `logbull`
4. Split on separator and take first part

### Multi-Repo Criteria

A package is considered multi-repo if:

- **2 or more** repositories match the pattern `<base><separator><ecosystem>`
- The repos belong to the same owner/organization
- The ecosystem suffixes are from the predefined list

## Example Output

```
================================================================================
Multi-Repo Detector (Parallel Processing)
================================================================================

GitHub Token Configuration
--------------------------------------------------------------------------------
✓ Loaded 3 token(s)
Using authenticated GitHub API (5,000 requests/hour per token)

Source directory: /path/to/Directory-Structure-Miner/results
Output directory: /path/to/Multirepo-Detector/results
Parallel workers: 10
Tokens available: 3

Processing 2_ecosystems directory...
  Reading Crates_Go.txt...
    Processing 100 packages in parallel (max 10 workers)...
    [1/100] logbull/logbull: MULTI-REPO (sep: -, repos: 8)
    [3/100] cosmos/cosmos-sdk: mono-repo
    [2/100] grpc/grpc: mono-repo
    [5/100] openai/openai: MULTI-REPO (sep: -, repos: 5)
    ...

Multi-repo packages saved to: results/multirepo_packages.csv
Summary saved to: results/summary.txt

================================================================================
Processing complete in 45.23 seconds!
================================================================================
Total packages analyzed: 5234
Multi-repo packages found: 42
Mono-repo packages: 5192

Results saved in: /path/to/results
```

**Note**: With parallel processing and 3 tokens, processing time reduced from ~245 seconds to ~45 seconds (5x speedup).

## Troubleshooting

### Rate Limit Exceeded

If you see rate limit errors:

1. Set `GITHUB_TOKEN` environment variable
2. Wait for the rate limit to reset
3. The script will automatically wait when approaching limits

### API Errors

If GitHub API returns errors:

- Check your internet connection
- Verify the GitHub token is valid
- Check if the repository/organization exists

### Missing Packages

If packages aren't detected:

- The organization might be private
- The naming pattern might not match standard conventions
- The repository might have been deleted or renamed
