# PHP/Packagist Miner

This tool downloads and processes the Packagist.org package list to extract PHP package information for cross-ecosystem analysis.

## Features

- Downloads package list from Packagist.org
- Fetches detailed metadata via Packagist API
- Extracts package metadata (ID, name, homepage, repository)
- Formats data for cross-ecosystem package analysis
- Progress tracking with visual feedback
- Rate-limited API calls to respect server resources
- Generates standardized CSV output compatible with Package-Filter

## Setup

### Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

This will:

- Create a virtual environment
- Install required dependencies (requests, tqdm)
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

Mine all PHP packages from Packagist.org:

```bash
source venv/bin/activate
python mine_php.py
```

Or run directly without activating:

```bash
venv/bin/python mine_php.py
```

The script will:

1. Download the list of all package names from Packagist.org
2. Fetch detailed information for each package via API
3. Generate CSV output in `Resource/Package/Package-List/PHP_New.csv`

### Data Sources

- **Package Names**: https://packagist.org/packages/list.json
- **Package Details**: https://packagist.org/packages/{vendor}/{package}.json
- **Format**: JSON

## Output Format

The script generates `PHP_New.csv` in the `Resource/Package/Package-List/` directory with the following structure:

```csv
ID,Platform,Name,Homepage URL,Repository URL
1,Packagist,symfony/symfony,https://symfony.com,https://github.com/symfony/symfony
2,Packagist,laravel/framework,https://laravel.com,https://github.com/laravel/framework
3,Packagist,guzzlehttp/guzzle,https://guzzlephp.org,https://github.com/guzzle/guzzle
```

### Column Descriptions

- **ID**: Sequential identifier (1, 2, 3, ...)
- **Platform**: Always "Packagist" for PHP packages
- **Name**: Package name as registered on Packagist.org (vendor/package format)
- **Homepage URL**: Project homepage (from package metadata)
- **Repository URL**: Source code repository URL

**Note**: This format is compatible with the Package-Filter tool for cross-ecosystem analysis.

## Processing Details

### API Rate Limiting

The script implements rate limiting to avoid overwhelming the Packagist API:

- **Rate**: 20 requests per second (0.05 second delay between requests)
- **Purpose**: Respectful API usage, avoiding server load
- **Impact**: Processing time increases with number of packages

**Estimated Time**: With ~400,000 packages and 20 req/sec, expect ~5-6 hours total runtime.

### Package Naming Convention

PHP packages follow the `vendor/package` naming pattern:

- `symfony/console`
- `laravel/framework`
- `doctrine/orm`

This two-part naming helps prevent conflicts and organize packages by maintainer.

### Package Metadata Sources

For each package, the script fetches:

```json
{
  "package": {
    "name": "symfony/console",
    "homepage": "https://symfony.com",
    "repository": "https://github.com/symfony/symfony",
    "versions": {
      "dev-master": {
        "source": {
          "url": "https://github.com/symfony/symfony.git",
          "type": "git"
        }
      }
    }
  }
}
```

The script prioritizes:

1. **Homepage**: `homepage` field → "nan"
2. **Repository**: `repository` field → version source URL → "nan"

### Repository URL Extraction

The script tries multiple strategies to find repository URLs:

1. **Direct Repository Field**: Uses `package.repository` if available
2. **Version Source**: Checks `dev-master`, `dev-main`, `master`, `main` branches
3. **First Version**: Falls back to first available version's source URL
4. **Validation**: Ensures URLs start with `http` or `https`

### Error Handling

If an API call fails (timeout, 404, etc.):

- Continues processing with "nan" values
- Logs no error (fails silently)
- Ensures complete dataset even with some missing data

## Files

- `mine_php.py`: Main script
- `requirements.txt`: Python dependencies (requests, tqdm)
- `setup.sh`: Automated setup script
- Output: `../../../Resource/Package/Package-List/PHP_New.csv`

## Troubleshooting

### "Error downloading package list"

Check that:

- You have internet connectivity
- packagist.org is accessible: `curl -I https://packagist.org/packages/list.json`
- No firewall blocking the connection

### Script is very slow

This is expected behavior:

- Rate limiting (20 requests/second) is intentional
- With 400K+ packages, expect 5-6 hours runtime
- Consider running overnight or in background

To run in background:

```bash
nohup python mine_php.py > output.log 2>&1 &
```

### "Error parsing JSON"

This can occur if:

- Packagist API response format changed
- Network corruption during download
- Server returned error page instead of JSON

**Solution**: Check internet connection and try again.

### "Permission denied" when creating output directory

Ensure you have write permissions to:

- Current directory (for temporary files)
- `Resource/Package/Package-List/` (for output)

### Incomplete data (many "nan" values)

This can occur if:

- API is temporarily unavailable
- Network issues during processing
- Some packages have incomplete metadata

**Note**: This is normal - not all packages have complete metadata on Packagist.org.

### Virtual environment issues

If you encounter errors related to the virtual environment:

1. Delete the `venv` folder: `rm -rf venv`
2. Re-run the setup script: `./setup.sh`
3. Virtual environments cannot be moved after creation - recreate if you move the directory

## Performance Notes

- **Download Time**: Fast (package list is relatively small JSON)
- **Processing Time**: SLOW (~5-6 hours for 400K+ packages)
- **Memory Usage**: Low (processes one package at a time)
- **Network Usage**: Moderate (many small API requests)

### Optimization Tips

To speed up processing (advanced users):

1. Reduce delay in `time.sleep(0.05)` (risks being rate-limited or blocked)
2. Use parallel requests (requires code modification)
3. Use Packagist metadata dump if available (check Packagist documentation)

## Advantages

- **Complete Data**: Includes all public packages
- **Official API**: Uses Packagist.org official endpoints
- **Detailed Metadata**: Gets homepage and repository URLs
- **Reliable**: Gracefully handles API failures
- **Rich Metadata**: Packagist provides comprehensive package information

## Limitations

- **Slow Processing**: Rate limiting means long runtime
- **API Dependent**: Requires Packagist.org to be available
- **Incomplete Metadata**: Not all packages have homepage/repository info
- **No Version Info**: Only captures general package information

## Alternative Approaches

For faster processing, consider:

1. **Packagist Dump**: Check if Packagist provides database dumps
2. **Metadata Files**: Some registries provide metadata files
3. **Cached Data**: Use previously downloaded data and update incrementally
4. **Parallel Processing**: Use async requests or multiprocessing (advanced)

---

## Code Explanation

### Architecture

The PHP Miner uses a two-phase approach:

1. **Phase 1**: Download complete list of package names
2. **Phase 2**: Fetch detailed metadata for each package

This mirrors the approach used by the Ruby miner, as both ecosystems provide similar API structures.

### 1. Package List Download

```python
packages_url = "https://packagist.org/packages/list.json"
data = response.json()
package_names = data.get('packageNames', [])
```

**Format**: JSON array of package names.

```json
{
  "packageNames": [
    "symfony/console",
    "laravel/framework",
    "guzzlehttp/guzzle",
    ...
  ]
}
```

**Advantages**:

- Fast download (single request)
- Complete list
- Simple JSON parsing

### 2. Package Metadata Fetching

```python
for package_name in package_names:
    time.sleep(0.05)  # Rate limiting (20 req/sec)
    response = requests.get(f"https://packagist.org/packages/{package_name}.json")
```

**Process**:

1. Iterate through each package name
2. Wait 0.05 seconds (rate limiting)
3. Fetch JSON metadata
4. Extract homepage and repository URLs
5. Write to CSV immediately (streaming write)

### 3. Complex Repository URL Extraction

```python
# Try direct repository field
repository = package_data.get('repository', '')

# Try version sources
versions = package_data.get('versions', {})
for version_key in ['dev-master', 'dev-main', 'master', 'main']:
    if version_key in versions:
        source = versions[version_key].get('source', {})
        repo_url = source.get('url', '')
```

**Strategy**: Multiple fallback levels.

**Why Complex**: Packagist stores repository info in multiple places:

- Direct `repository` field (not always present)
- Version-specific source URLs (most reliable)
- Different branch naming conventions (master vs main)

### 4. Version Priority

```python
for version_key in ['dev-master', 'dev-main', 'master', 'main']:
```

**Priority Order**:

1. `dev-master` (most common development branch)
2. `dev-main` (newer naming convention)
3. `master` (tagged version)
4. `main` (tagged version)

**Fallback**: If none found, use first available version.

### 5. URL Validation

```python
if homepage_url and not homepage_url.startswith('http'):
    homepage_url = "nan"
if repo_url and not repo_url.startswith('http'):
    repo_url = "nan"
```

**Purpose**: Filter out invalid URLs.

- Some packages have placeholder text instead of URLs
- Ensures data quality
- "nan" represents missing/invalid data

### 6. Streaming Write

```python
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for package_name in packages:
        # Fetch data
        writer.writerow([...])  # Write immediately
```

**Advantages**:

- Low memory usage (doesn't store all data in memory)
- Progress saved even if script crashes
- Can resume partially completed runs (with modification)

---

## Data Quality Notes

### Repository URL Accuracy

Packagist repository URLs are generally high quality because:

- Composer (PHP package manager) requires this information
- Most packages are hosted on GitHub
- Package authors maintain metadata actively

### Missing Data Patterns

Common reasons for "nan" values:

1. **Abandoned Packages**: No longer maintained, incomplete metadata
2. **Private Packages**: Listed but not publicly accessible
3. **Vanity URLs**: Homepage set to packagist.org page itself
4. **Legacy Packages**: Created before Packagist required full metadata

---
