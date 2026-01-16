# Ruby Gems Miner

This tool downloads and processes the RubyGems.org package list to extract Ruby gem information for cross-ecosystem analysis.

## Features

- Downloads gem list from RubyGems.org
- Fetches detailed metadata via RubyGems API
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

Mine all Ruby gems from RubyGems.org:

```bash
source venv/bin/activate
python mine_ruby.py
```

Or run directly without activating:

```bash
venv/bin/python mine_ruby.py
```

The script will:

1. Download the compact specs file (specs.4.8.gz)
2. Download the list of all gem names from RubyGems.org
3. Fetch detailed information for each gem via API
4. Generate CSV output in `Resource/Package/Package-List/Ruby_New.csv`
5. Clean up temporary files

### Data Sources

- **Gem Names**: http://rubygems.org/names
- **Gem Details**: https://rubygems.org/api/v1/gems/{name}.json
- **Format**: Plain text list (names) and JSON (details)

## Output Format

The script generates `Ruby_New.csv` in the `Resource/Package/Package-List/` directory with the following structure:

```csv
ID,Platform,Name,Homepage URL,Repository URL
1,RubyGems,rails,https://rubyonrails.org,https://github.com/rails/rails
2,RubyGems,devise,https://github.com/heartcombo/devise,https://github.com/heartcombo/devise
3,RubyGems,rake,https://github.com/ruby/rake,https://github.com/ruby/rake
```

### Column Descriptions

- **ID**: Sequential identifier (1, 2, 3, ...)
- **Platform**: Always "RubyGems" for Ruby packages
- **Name**: Gem name as registered on RubyGems.org
- **Homepage URL**: Project homepage (from gem metadata)
- **Repository URL**: Source code repository URL

**Note**: This format is compatible with the Package-Filter tool for cross-ecosystem analysis.

## Processing Details

### API Rate Limiting

The script implements rate limiting to avoid overwhelming the RubyGems API:

- **Rate**: 10 requests per second (0.1 second delay between requests)
- **Purpose**: Respectful API usage, avoiding server load
- **Impact**: Processing time increases with number of gems

**Estimated Time**: With ~180,000 gems and 10 req/sec, expect ~5 hours total runtime.

### Gem Metadata Sources

For each gem, the script fetches:

```json
{
  "name": "rails",
  "homepage_uri": "https://rubyonrails.org",
  "source_code_uri": "https://github.com/rails/rails",
  "project_uri": "https://rubygems.org/gems/rails"
}
```

The script prioritizes:

1. **Homepage**: `homepage_uri` → `project_uri` → "nan"
2. **Repository**: `source_code_uri` → `homepage_uri` → "nan"

### Error Handling

If an API call fails (timeout, 404, etc.):

- Continues processing with "nan" values
- Logs no error (fails silently)
- Ensures complete dataset even with some missing data

## Files

- `mine_ruby.py`: Main script
- `requirements.txt`: Python dependencies (requests, tqdm)
- `setup.sh`: Automated setup script
- `specs.4.8.gz`: Temporary download file (deleted after use)
- `specs.4.8`: Temporary decompressed file (deleted after use)
- Output: `../../../Resource/Package/Package-List/Ruby_New.csv`

## Troubleshooting

### "Error downloading gem names"

Check that:

- You have internet connectivity
- rubygems.org is accessible: `curl -I http://rubygems.org/names`
- No firewall blocking the connection

### Script is very slow

This is expected behavior:

- Rate limiting (10 requests/second) is intentional
- With 180K+ gems, expect 5+ hours runtime
- Consider running overnight or in background

To run in background:

```bash
nohup python mine_ruby.py > output.log 2>&1 &
```

### "Permission denied" when creating output directory

Ensure you have write permissions to:

- Current directory (for temporary files)
- `Resource/Package/Package-List/` (for output)

### Incomplete data (many "nan" values)

This can occur if:

- API is temporarily unavailable
- Network issues during processing
- Some gems have incomplete metadata

**Note**: This is normal - not all gems have complete metadata on RubyGems.org.

### Virtual environment issues

If you encounter errors related to the virtual environment:

1. Delete the `venv` folder: `rm -rf venv`
2. Re-run the setup script: `./setup.sh`
3. Virtual environments cannot be moved after creation - recreate if you move the directory

## Performance Notes

- **Download Time**: Fast (gem names list is small)
- **Processing Time**: SLOW (~5 hours for 180K+ gems)
- **Memory Usage**: Low (processes one gem at a time)
- **Network Usage**: Moderate (many small API requests)

### Optimization Tips

To speed up processing (advanced users):

1. Reduce delay in `time.sleep(0.1)` (risks being rate-limited or blocked)
2. Use parallel requests (requires code modification)
3. Use RubyGems database dump instead of API (requires parsing Marshal format)

## Advantages

- **Complete Data**: Includes all public gems
- **Official API**: Uses RubyGems.org official endpoints
- **Detailed Metadata**: Gets homepage and repository URLs
- **Reliable**: Gracefully handles API failures

## Limitations

- **Slow Processing**: Rate limiting means long runtime
- **API Dependent**: Requires RubyGems.org to be available
- **Incomplete Metadata**: Not all gems have homepage/repository info
- **No Version Info**: Only captures latest gem information

## Alternative Approaches

For faster processing, consider:

1. **Database Dump**: RubyGems provides database dumps (requires PostgreSQL)
2. **Bulk API**: Some bulk endpoints may exist (check RubyGems API docs)
3. **Cached Data**: Use previously downloaded data and update incrementally

---

## Code Explanation

### Architecture

The Ruby Miner uses a two-phase approach:

1. **Phase 1**: Download complete list of gem names
2. **Phase 2**: Fetch detailed metadata for each gem

This is necessary because RubyGems doesn't provide a single complete dump like crates.io.

### 1. Specs File Download

```python
specs_url = "https://rubygems.org/specs.4.8.gz"
download_file(specs_url, specs_gz_path)
```

**Purpose**: Downloads compact specs (currently not parsed, but available for future use).

**Note**: The specs file is in Ruby Marshal format (binary), which is complex to parse in Python. Currently, we use the simpler names endpoint instead.

### 2. Gem Names List

```python
names_url = "http://rubygems.org/names"
gem_names = response.text.strip().split('\n')
```

**Format**: Simple newline-delimited text file.

```
rails
devise
rake
...
```

**Advantages**:

- Simple to parse
- Complete list of all gems
- Fast download

### 3. Detailed Metadata Fetching

```python
for gem_name in gem_names:
    time.sleep(0.1)  # Rate limiting
    response = requests.get(f"https://rubygems.org/api/v1/gems/{gem_name}.json")
```

**Process**:

1. Iterate through each gem name
2. Wait 0.1 seconds (rate limiting)
3. Fetch JSON metadata
4. Extract homepage and repository URLs
5. Write to CSV immediately (streaming write)

### 4. URL Extraction Priority

```python
homepage_url = gem_info.get('homepage_uri', '') or gem_info.get('project_uri', '') or "nan"
repo_url = gem_info.get('source_code_uri', '') or gem_info.get('homepage_uri', '') or "nan"
```

**Fallback Chain**:

- **Homepage**: Try `homepage_uri` first, fall back to `project_uri`
- **Repository**: Try `source_code_uri` first, fall back to `homepage_uri`

**Validation**:

```python
if homepage_url and not homepage_url.startswith('http'):
    homepage_url = "nan"
```

Ensures only valid HTTP/HTTPS URLs are kept.

---
