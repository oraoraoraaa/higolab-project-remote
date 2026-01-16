# NPM Package Miner

This tool mines the npm registry to collect information about all npm packages.

## Features

- Fetches complete list of npm packages from the official npm registry
- Retrieves package metadata including homepage and repository URLs via npm registry API
- Parallel processing with 50 workers for efficient data collection
- Handles various repository URL formats (git+https, git@github, etc.)
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
python mine_npm.py
```

The script will:

1. Download the complete list of package names from npm registry (~2-3 million packages)
2. Fetch detailed metadata for each package in parallel
3. Save results to `../../../Resource/Package/Package-List/NPM.csv`

## Output Format

CSV file with columns:

- `ID`: Sequential package identifier
- `Platform`: "NPM"
- `Name`: Package name
- `Homepage URL`: Package homepage URL (from package.json)
- `Repository URL`: Source code repository URL (normalized to HTTPS format)

## Data Source

- **Registry**: https://registry.npmjs.org/
- **All packages list**: https://replicate.npmjs.com/_all_docs
- **Package metadata**: https://registry.npmjs.org/{package-name}

## Performance

- Expected runtime: 10-20 hours for ~2-3 million packages
- 50 parallel workers for API requests
- Network-dependent (typically limited by API rate and network speed)

## Notes

- The npm registry is continuously updated, so package counts may vary
- Repository URLs are normalized to HTTPS format
- Missing or invalid URLs are marked as "nan"
- The script handles API errors gracefully and continues processing
