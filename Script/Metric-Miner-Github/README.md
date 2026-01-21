# GitHub Metrics Miner

Mines GitHub repository metrics from Config-Locator results and outputs them as CSV files.

## Features

- Extracts comprehensive GitHub metrics:
  - Stars
  - Commits count
  - Pull requests (active, closed, all)
  - Issues (active, closed, all)
  - Contributors count
- Automatically skips forked repositories
- Supports multiple GitHub tokens with automatic rotation
- Token validation and rate limit monitoring
- Organizes output by ecosystem count
- Progress tracking with detailed status updates
- Generates summary report

## Setup

Run the setup script to create a virtual environment and install dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

Or manually install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic Usage

Activate the virtual environment and run the script:

```bash
source venv/bin/activate
python mine_metrics_github.py
```

The script will prompt you to enter GitHub token(s). You can enter multiple tokens separated by commas.

### Command Line Options

```bash
python mine_metrics_github.py [OPTIONS]
```

**Options:**

- `--input-dir DIR`: Directory containing txt input files (default: `../Config-Locator/results`)
- `--output-dir DIR`: Directory to save output CSV files (default: `results`)
- `--token TOKEN [TOKEN ...]`: GitHub personal access token(s)
- `--files FILE [FILE ...]`: Specific txt files to process (default: all txt files)

### Examples

**Using tokens from command line:**

```bash
python mine_metrics_github.py --token ghp_xxxxxxxxxxxx
```

**Multiple tokens:**

```bash
python mine_metrics_github.py --token ghp_token1 ghp_token2 ghp_token3
```

**Process specific files:**

```bash
python mine_metrics_github.py --files Crates_Go.txt Maven_PyPI.txt
```

**Custom directories:**

```bash
python mine_metrics_github.py --input-dir /path/to/input --output-dir /path/to/output
```

## GitHub Token

### Getting a Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `public_repo` (or `repo` for private repos)
4. Generate and copy the token

### Using Multiple Tokens

Multiple tokens are recommended for large-scale mining to avoid rate limits:

- Each token has 5,000 requests/hour
- The script automatically rotates to the next token when one is depleted
- Enter tokens separated by commas when prompted

### Environment Variable

You can also set tokens via environment variable:

```bash
export GITHUB_TOKEN=ghp_token1,ghp_token2,ghp_token3
python mine_metrics_github.py
```

## Input Format

The script processes txt files from Config-Locator with the following structure:

```
Package 1/51
================================================================================
Name: wrpc
Repository: https://github.com/bytecodealliance/wrpc
Owner/Repo: bytecodealliance/wrpc
Expected Ecosystems: Crates, Go
...
```

## Output Format

### CSV Files

Output CSV files contain the following columns:

| Column                 | Description             |
| ---------------------- | ----------------------- |
| `name`                 | Package name            |
| `repo_url`             | GitHub repository URL   |
| `stars`                | Number of stars         |
| `commits`              | Total commit count      |
| `active_pull_requests` | Number of open PRs      |
| `closed_pull_requests` | Number of closed PRs    |
| `all_pull_requests`    | Total PRs               |
| `contributors`         | Number of contributors  |
| `active_issues`        | Number of open issues   |
| `closed_issues`        | Number of closed issues |
| `all_issues`           | Total issues            |

### Directory Structure

Output files are organized by ecosystem count:

```
results/
├── 2_ecosystems/
│   ├── Crates_Go.csv
│   ├── Maven_PyPI.csv
│   └── ...
├── 3_ecosystems/
│   ├── Crates_Go_Maven.csv
│   └── ...
├── 4_ecosystems/
│   └── ...
└── summary.txt
```

### Summary File

A `summary.txt` file is generated with statistics:

- Processing statistics by ecosystem count
- Success rates
- Detailed results for each file
- List of skipped files

## Features

### Fork Detection

The script automatically skips repositories that are forks to ensure you're only collecting metrics from original repositories.

### Token Management

- **Validation**: Checks all tokens before processing
- **Rotation**: Automatically switches tokens when rate limit is low
- **Status Display**: Shows remaining requests and reset times

### Rate Limiting

- Checks rate limit every 10 packages
- Automatically rotates to next token when current is depleted
- Displays warnings when running low on requests

### Error Handling

- Handles network errors gracefully
- Skips repositories that are not found or inaccessible
- Continues processing even if individual packages fail
- Provides detailed error messages

## Rate Limits

- **With token**: 5,000 requests per hour per token
- **Without token**: 60 requests per hour (not recommended)

Each package requires approximately 5-8 API requests, so:

- With 1 token: ~625-1000 packages per hour
- With 3 tokens: ~1875-3000 packages per hour

## Tips

1. **Use multiple tokens** for large datasets to avoid rate limit delays
2. **Run during off-peak hours** for better API responsiveness
3. **Process incrementally** using `--files` to handle errors more easily
4. **Check summary.txt** after completion to verify results

## Troubleshooting

### Rate Limit Exceeded

If you see rate limit errors:

- Add more tokens using `--token` option
- Wait for rate limit to reset (check reset time in token status)
- Process files incrementally

### Repository Not Found

Some repositories may be:

- Deleted
- Made private
- Renamed
- Invalid URLs

These are automatically skipped and logged.

### Connection Errors

If you encounter connection issues:

- Check your internet connection
- Verify GitHub API status: https://www.githubstatus.com/
- Try again later

## Deactivate

When done, deactivate the virtual environment:

```bash
deactivate
```
