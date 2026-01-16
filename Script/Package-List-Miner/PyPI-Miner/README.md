# PyPI Package Miner

This tool mines the PyPI (Python Package Index) to collect information about all Python packages.

## Features

- Fetches complete list of PyPI packages from the official simple API
- Retrieves package metadata including homepage and repository URLs via PyPI JSON API
- Parallel processing with 50 workers for efficient data collection
- Intelligently extracts repository URLs from multiple metadata fields
- Progress tracking with visual feedback
- Outputs to CSV format compatible with cross-ecosystem analysis

## Setup

### Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

This will:

- Create a virtual environment
- Install required dependencies (requests, tqdm)

### Manual Setup (Alternative)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
source .venv/bin/activate
python mine_pypi.py
```

The script will:

1. Download the complete list of package names from PyPI simple index (~500k packages)
2. Fetch detailed metadata for each package in parallel
3. Save results to `../../../Resource/Package/Package-List/PyPI.csv`

## Output Format

CSV file with columns:

- `ID`: Sequential package identifier
- `Platform`: "PyPI"
- `Name`: Package name
- `Homepage URL`: Package homepage URL (from package metadata)
- `Repository URL`: Source code repository URL (extracted from project_urls or home_page)

## Data Source

- **Simple Index**: https://pypi.org/simple/
- **Package metadata**: https://pypi.org/pypi/{package-name}/json

## Repository URL Detection

The script intelligently searches for repository URLs in the following order:

1. `project_urls` field with keys: Source, Source Code, Repository, Code, GitHub, GitLab
2. `home_page` field if it contains github.com, gitlab.com, or bitbucket.org

## Performance

- Expected runtime: 3-8 hours for ~500k packages
- 50 parallel workers for API requests
- Network-dependent (typically limited by API rate and network speed)

## Notes

- PyPI is continuously updated, so package counts may vary
- Repository URLs are validated to start with http/https
- Missing or invalid URLs are marked as "nan"
- The script handles API errors gracefully and continues processing
