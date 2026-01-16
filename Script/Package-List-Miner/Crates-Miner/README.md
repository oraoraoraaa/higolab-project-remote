# Crates.io Package Miner

This tool downloads and processes the complete crates.io database dump to extract Rust package information for cross-ecosystem analysis.

## Features

- Downloads the official crates.io database dump
- Extracts package metadata (ID, name, homepage, repository)
- Formats data for cross-ecosystem package analysis
- Progress tracking with visual feedback
- Automatic cleanup of temporary files
- Generates standardized CSV output compatible with Package-Filter

## Setup

### Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

This will:

- Create a virtual environment
- Install required dependencies (requests, pandas, tqdm)
- Prepare the environment for mining

### Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Important**: Virtual environments contain hardcoded paths and **cannot be moved** after creation. If you need to relocate this script:

1. Delete the `venv` folder
2. Recreate it in the new location
3. Reinstall the packages

## Usage

Mine all crates from the crates.io database:

```bash
source venv/bin/activate
python mine_crates.py
```

Or run directly without activating:

```bash
venv/bin/python mine_crates.py
```

The script will:

1. Download the crates.io database dump (~1000+ MB)
2. Extract the archive
3. Process crate metadata
4. Generate CSV output in `Resource/Package/Package-List/Crates_New.csv`
5. Clean up temporary files

### What Gets Downloaded

- **Source**: `https://static.crates.io/db-dump.tar.gz`
- **Size**: ~1000 MB (compressed)
- **Contents**: Complete crates.io database snapshot
- **Format**: TAR.GZ archive containing CSV files

## Input Format

The crates.io database dump contains several CSV files. This script uses `crates.csv` which includes:

- `id`: Unique crate identifier
- `name`: Crate name
- `homepage`: Homepage URL (may be empty)
- `repository`: Source code repository URL (may be empty)
- Additional metadata (description, downloads, etc.)

## Output Format

The script generates `Crates_New.csv` in the `Resource/Package/Package-List/` directory with the following structure:

```csv
ID,Platform,Name,Homepage URL,Repository URL
1,Crates.io,rand,https://rust-random.github.io/book,https://github.com/rust-random/rand
2,Crates.io,serde,https://serde.rs,https://github.com/serde-rs/serde
3,Crates.io,tokio,https://tokio.rs,https://github.com/tokio-rs/tokio
```

### Column Descriptions

- **ID**: Sequential identifier (1, 2, 3, ...)
- **Platform**: Always "Crates.io" for Rust packages
- **Name**: Crate name as registered on crates.io
- **Homepage URL**: Project homepage (from crate metadata)
- **Repository URL**: Source code repository URL (typically GitHub)

**Note**: This format is compatible with the Package-Filter tool for cross-ecosystem analysis.

## Processing Details

### Database Dump Structure

The downloaded archive contains a dated directory (e.g., `2025-11-03-020107`) with:

```
2025-11-03-020107/
├── data/
│   ├── crates.csv           ← Used by this script
│   ├── versions.csv
│   ├── dependencies.csv
│   ├── teams.csv
│   └── ... (other files)
└── metadata/
    └── ... (metadata files)
```

### Extraction Process

1. **Download**: Fetches `db-dump.tar.gz` from static.crates.io
2. **Extract**: Decompresses to `Code/Script/Crates-Miner/crates-db/`
3. **Process**: Reads `data/crates.csv` from the dated directory
4. **Transform**: Converts to standardized format
5. **Output**: Writes to `Resource/Package/Package-List/Crates_New.csv`
6. **Cleanup**: Deletes the `.tar.gz` file (keeps extracted data)

### Data Transformation

The script transforms crates.io data to match the cross-ecosystem format:

**Input** (crates.csv):

```csv
id,name,homepage,repository,description,downloads,...
12345,serde,https://serde.rs,https://github.com/serde-rs/serde,"A serialization framework",50000000,...
```

**Output** (Crates_New.csv):

```csv
ID,Platform,Name,Homepage URL,Repository URL
1,Crates.io,serde,https://serde.rs,https://github.com/serde-rs/serde
```

## Files

- `mine_crates.py`: Main script
- `requirements.txt`: Python dependencies (requests, pandas, tqdm)
- `setup.sh`: Automated setup script
- `crates-db/`: Temporary directory for extracted database (created during execution)
- Output: `../../../Resource/Package/Package-List/Crates_New.csv`

## Troubleshooting

### "Could not download database dump"

Check that:

- You have internet connectivity
- crates.io is accessible: `curl -I https://static.crates.io/db-dump.tar.gz`
- You have sufficient disk space (~500 MB for extraction)

### "crates.csv not found"

This may occur if:

- The database dump structure has changed
- Extraction failed partway through
- The archive is corrupted

**Solution**: Delete `crates-db/` directory and run again to re-download.

### "Permission denied" when creating output directory

Ensure you have write permissions to:

- Current directory (for temporary files)
- `Resource/Package/Package-List/` (for output)

### "Memory error" during processing

The crates.io database is large (100K+ crates). If you encounter memory issues:

- Close other applications
- Increase available system memory
- Consider processing in chunks (requires script modification)

### Virtual environment issues

If you encounter errors related to the virtual environment:

1. Delete the `venv` folder: `rm -rf venv`
2. Re-run the setup script: `./setup.sh`
3. Virtual environments cannot be moved after creation - recreate if you move the directory

---

## Code Explanation

### Architecture Overview

The Crates.io Miner is a simpler tool compared to the Directory Structure Miners, focused on a single task: downloading and processing the official crates.io database dump.

Key characteristics:

- **Batch Processing**: Downloads entire database at once
- **Official Source**: Uses crates.io's public database dump
- **No API Calls**: Works with static data dump (no rate limiting)
- **Large Scale**: Processes 100K+ packages

### 1. Download Function

```python
def download_file(url, filename):
    """Downloads a file from a URL with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte

    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)
```

**Purpose**: Downloads large files with progress tracking.

**Features**:

- **Streaming**: Uses `stream=True` to avoid loading entire file in memory
- **Progress Bar**: Shows download progress with `tqdm`
- **Chunk Processing**: Downloads in 1KB chunks
- **Size Display**: Shows human-readable units (MB, GB)

**Process**:

1. Make HTTP GET request with streaming
2. Get total file size from headers
3. Open file in binary write mode
4. Download and write in chunks
5. Update progress bar after each chunk

### 2. Main Mining Function

```python
def mine_crates():
    """Mines crates.io to get the whole list of Rust packages from the database dump."""

    dump_url = "https://static.crates.io/db-dump.tar.gz"
    dump_path = "db-dump.tar.gz"
    extract_path = "Code/Script/Crates-Miner/crates-db"
```

**Purpose**: Orchestrates the entire mining process.

**Configuration**:

- **dump_url**: Official crates.io database dump URL
- **dump_path**: Local filename for downloaded archive
- **extract_path**: Directory for extracted files

### 3. Download Phase

```python
    # Download the database dump
    if not os.path.exists(dump_path):
        print("Downloading crates.io database dump...")
        download_file(dump_url, dump_path)
    else:
        print("Database dump already downloaded.")
```

**Logic**:

- Checks if archive already exists
- Skips download if file present (saves time on reruns)
- Downloads ~100-200 MB compressed file

**Why This Matters**:

- Avoids re-downloading large file unnecessarily
- Useful during development/testing
- Saves bandwidth and time

### 4. Extraction Phase

```python
    # Extract the database dump
    print("Extracting database dump...")
    with tarfile.open(dump_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

    # Delete the tar.gz file
    if os.path.exists(dump_path):
        print("Deleting database dump archive...")
        os.remove(dump_path)
```

**Process**:

1. **Open Archive**: Opens .tar.gz file in read mode
2. **Extract All**: Extracts to `crates-db/` directory
3. **Cleanup**: Removes archive to save disk space

**Tar Format**: `r:gz` means read mode with gzip compression

### 5. Directory Discovery

```python
    # Find the actual data directory (it has a date in the name)
    data_dir = ""
    for item in os.listdir(extract_path):
        if os.path.isdir(os.path.join(extract_path, item)):
            data_dir = os.path.join(extract_path, item)
            break

    if not data_dir:
        print("Could not find data directory in the extracted archive.")
        return
```

**Purpose**: Finds the dated directory containing the actual data.

**Why Dynamic**: The database dump directory name changes with each snapshot:

- `2025-11-03-020107/`
- `2025-11-04-020107/`
- etc.

**Logic**:

1. List all items in extraction path
2. Find first directory (not file)
3. Use that as data directory
4. Error if no directory found

### 6. CSV Path Construction

```python
    crates_csv_path = os.path.join(data_dir, "data", "crates.csv")
    if not os.path.exists(crates_csv_path):
        print(f"crates.csv not found in {data_dir}")
        return
```

**Path Structure**: `crates-db/{date}/data/crates.csv`

**Validation**: Checks file exists before attempting to read

### 7. Data Processing

```python
    print("Processing crate data...")
    # Read the crates data
    df = pd.read_csv(crates_csv_path)

    # Create the path to the output file
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Resource', 'Package', 'Package-List'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "Crates_New.csv")
```

**Reading**:

- Uses pandas `read_csv()` for efficient processing
- Automatically handles CSV parsing and data types

**Output Path**:

- Navigates up 3 directories from script location
- Ensures output directory exists (`makedirs`)
- Constructs full path to output file

**Path Navigation**:

```
Current: Code/Script/Crates-Miner/mine_crates.py
Up 1:    Code/Script/Crates-Miner/
Up 2:    Code/Script/
Up 3:    Code/
Result:  Code/../../../Resource/Package/Package-List/
```

### 8. CSV Writing

```python
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Platform", "Name", "Homepage URL", "Repository URL"])

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Writing to CSV"):
            writer.writerow([
                index + 1,
                "Crates.io",
                row["name"],
                row["homepage"],
                row["repository"],
            ])

    print(f"Successfully saved {df.shape[0]} crates to {output_file}")
```

**Process**:

1. **Open File**: In write mode with UTF-8 encoding
2. **Write Header**: Column names for standardized format
3. **Iterate Rows**: Loop through pandas DataFrame with progress bar
4. **Transform Data**: Convert each row to standard format
5. **Write Row**: Add to output CSV

**Data Transformation**:

- **ID**: Uses index + 1 (1-based instead of 0-based)
- **Platform**: Hardcoded as "Crates.io"
- **Name**: Direct mapping from `row["name"]`
- **Homepage URL**: Direct mapping from `row["homepage"]`
- **Repository URL**: Direct mapping from `row["repository"]`

**Progress Tracking**:

- `tqdm()` with `total=df.shape[0]` shows percentage complete
- Useful for large datasets (100K+ rows)
- Provides time estimates

### Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    Crates.io Miner Workflow                 │
└─────────────────────────────────────────────────────────────┘

1. Check if db-dump.tar.gz exists
   ├─ Yes → Skip download
   └─ No  → Download from static.crates.io (~200 MB)

2. Extract db-dump.tar.gz
   ├─ Decompress with gzip
   ├─ Extract tar to crates-db/
   └─ Delete archive file

3. Find dated directory
   ├─ Scan crates-db/
   └─ Locate {date}/data/crates.csv

4. Load data with pandas
   └─ Read entire crates.csv into DataFrame

5. Create output directory structure
   └─ Resource/Package/Package-List/

6. Transform and write CSV
   ├─ Header: ID, Platform, Name, Homepage URL, Repository URL
   ├─ For each crate:
   │  ├─ Generate sequential ID
   │  ├─ Set Platform = "Crates.io"
   │  ├─ Copy name, homepage, repository
   │  └─ Write row
   └─ Show progress bar

7. Complete
   └─ Print success message with count
```

### Error Handling

The script includes basic error handling for common scenarios:

**Download Errors**:

```python
response.raise_for_status()
```

- Raises exception for HTTP errors (404, 500, etc.)
- Stops execution if download fails

**File Not Found**:

```python
if not os.path.exists(crates_csv_path):
    print(f"crates.csv not found in {data_dir}")
    return
```

- Checks for expected files
- Gracefully exits with error message

**Directory Creation**:

```python
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

- Creates output directory if missing
- Prevents write errors

---
