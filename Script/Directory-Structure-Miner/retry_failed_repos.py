#!/usr/bin/env python3
"""
Retry Failed Repository Mining
Retries all failed repositories from error logs and updates output files and summary.
"""

import os
import re
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# Import the miner from the main script
from mine_directory_structure import (
    GitHubDirectoryMiner,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ERROR_LOG_DIR,
    DATASET_DIR
)

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
RETRY_ERROR_LOG_DIR = DATASET_DIR / "Directory-Structure-Miner/retry-error-log"


def parse_error_log(log_file_path: str) -> List[Dict[str, str]]:
    """
    Parse an error log file and extract failed repository information.
    
    Args:
        log_file_path: Path to the error log file
        
    Returns:
        List of dictionaries containing repository info and error details
    """
    errors = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by error entries
    error_blocks = re.split(r'Error #\d+\n-+\n', content)[1:]  # Skip header
    
    for block in error_blocks:
        error_info = {}
        
        # Extract timestamp
        timestamp_match = re.search(r'Timestamp: (.+)', block)
        if timestamp_match:
            error_info['timestamp'] = timestamp_match.group(1)
        
        # Extract repository URL
        repo_match = re.search(r'Repository: (.+)', block)
        if repo_match:
            error_info['repository'] = repo_match.group(1).strip()
        
        # Extract ecosystems
        ecosystems_match = re.search(r'Ecosystems: (.+)', block)
        if ecosystems_match:
            error_info['ecosystems'] = [e.strip() for e in ecosystems_match.group(1).split(',')]
        
        # Extract error type
        error_type_match = re.search(r'Error Type: (.+)', block)
        if error_type_match:
            error_info['error_type'] = error_type_match.group(1).strip()
        
        # Extract error message
        error_msg_match = re.search(r'Error Message: (.+)', block)
        if error_msg_match:
            error_info['error_message'] = error_msg_match.group(1).strip()
        
        if error_info.get('repository'):
            errors.append(error_info)
    
    return errors


def get_all_error_logs(error_log_dir: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Get all error logs from the error log directory.
    
    Args:
        error_log_dir: Directory containing error log files
        
    Returns:
        Dictionary mapping ecosystem combination to list of error entries
    """
    error_log_path = Path(error_log_dir)
    all_errors = {}
    
    for log_file in error_log_path.glob("*_errors.log"):
        # Extract ecosystem combination from filename
        ecosystem_combination = log_file.stem.replace("_errors", "")
        
        # Parse the log file
        errors = parse_error_log(str(log_file))
        
        if errors:
            all_errors[ecosystem_combination] = errors
    
    return all_errors


def append_to_output_file(
    output_file_path: str,
    repo_url: str,
    owner: str,
    repo: str,
    ecosystems: List[str],
    tree_output: str,
    current_count: int,
    new_count: int
):
    """
    Append a successfully mined repository to the output file.
    
    Args:
        output_file_path: Path to the output file
        repo_url: Repository URL
        owner: Repository owner
        repo: Repository name
        ecosystems: List of ecosystems
        tree_output: Directory tree structure
        current_count: Current count before adding
        new_count: New count after adding
    """
    with open(output_file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Package {new_count}/{new_count}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Repository: {repo_url}\n")
        f.write(f"Owner/Repo: {owner}/{repo}\n")
        f.write(f"Ecosystems: {', '.join(ecosystems)}\n\n")
        f.write(f"Directory Structure:\n")
        f.write(f"{'-' * 80}\n\n")
        f.write(tree_output)
        f.write(f"\n\n{'=' * 80}\n")


def update_output_file_header(
    output_file_path: str,
    old_mined_count: int,
    new_mined_count: int,
    total_count: int
):
    """
    Update the header of an output file with new counts.
    
    Args:
        output_file_path: Path to the output file
        old_mined_count: Previous successfully mined count
        new_mined_count: New successfully mined count
        total_count: Total package count
    """
    with open(output_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update the header
    content = re.sub(
        r'Successfully Mined: \d+',
        f'Successfully Mined: {new_mined_count}',
        content,
        count=1
    )
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def retry_failed_repos(
    error_log_dir: str,
    output_dir: str,
    miner: GitHubDirectoryMiner,
    max_depth: Optional[int] = None
):
    """
    Retry all failed repositories and update output files.
    
    Args:
        error_log_dir: Directory containing error logs
        output_dir: Directory containing output files
        miner: GitHubDirectoryMiner instance
        max_depth: Maximum depth to traverse
    """
    print(f"\n{'=' * 80}")
    print("Retry Failed Repository Mining")
    print(f"{'=' * 80}")
    print(f"Error log directory: {error_log_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")
    
    # Get all error logs
    print("Parsing error logs...")
    all_errors = get_all_error_logs(error_log_dir)
    
    if not all_errors:
        print("No error logs found!")
        return
    
    # Count total errors and unique repositories
    total_errors = sum(len(errors) for errors in all_errors.values())
    unique_repos = set()
    for errors in all_errors.values():
        for error in errors:
            unique_repos.add(error['repository'])
    
    print(f"Total error entries: {total_errors}")
    print(f"Unique repositories to retry: {len(unique_repos)}")
    print(f"Ecosystem combinations with errors: {len(all_errors)}")
    
    # Group by ecosystem combination
    print(f"\nErrors by ecosystem combination:")
    for combo, errors in sorted(all_errors.items()):
        print(f"  {combo}: {len(errors)} errors")
    
    # Ask for confirmation
    print(f"\n{'=' * 80}")
    try:
        response = input("Proceed with retry? [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            print("Retry cancelled.")
            return
    except (KeyboardInterrupt, EOFError):
        print("\nRetry cancelled.")
        return
    
    # Process each ecosystem combination
    print(f"\n{'=' * 80}")
    print("Starting retry process...")
    print(f"{'=' * 80}\n")
    
    retry_summary = {}
    
    for ecosystem_combination, errors in sorted(all_errors.items()):
        print(f"\n{'=' * 80}")
        print(f"Retrying: {ecosystem_combination}")
        print(f"{'=' * 80}")
        print(f"Total errors: {len(errors)}")
        
        # Determine output file path
        ecosystems = ecosystem_combination.split('_')
        ecosystem_count = len(ecosystems)
        output_subdir = Path(output_dir) / f"{ecosystem_count}_ecosystems"
        output_file = output_subdir / f"{ecosystem_combination}.txt"
        
        if not output_file.exists():
            print(f"  ⚠ Output file not found: {output_file}")
            continue
        
        # Read current counts from output file
        with open(output_file, 'r', encoding='utf-8') as f:
            header = f.read(1000)
        
        total_match = re.search(r'Total Packages: (\d+)', header)
        mined_match = re.search(r'Successfully Mined: (\d+)', header)
        
        if not total_match or not mined_match:
            print(f"  ⚠ Could not parse counts from output file")
            continue
        
        total_packages = int(total_match.group(1))
        current_mined = int(mined_match.group(1))
        
        print(f"  Current: {current_mined}/{total_packages} mined")
        
        # Retry each failed repository
        successful_retries = 0
        failed_retries = 0
        
        with tqdm(total=len(errors), desc=f"  Retrying {ecosystem_combination}", unit="repo") as pbar:
            for error in errors:
                repo_url = error['repository']
                ecosystems_list = error['ecosystems']
                
                # Parse GitHub URL
                parsed = miner.parse_github_url(repo_url)
                if not parsed:
                    failed_retries += 1
                    pbar.update(1)
                    continue
                
                owner, repo = parsed
                
                # Try to get the directory tree
                tree_output = miner.get_tree(
                    owner=owner,
                    repo=repo,
                    max_depth=max_depth,
                    repo_url=repo_url,
                    ecosystems=ecosystems_list,
                    ecosystem_combination=ecosystem_combination,
                    retry_count=0
                )
                
                if tree_output:
                    # Success! Append to output file
                    new_mined = current_mined + successful_retries + 1
                    append_to_output_file(
                        output_file_path=str(output_file),
                        repo_url=repo_url,
                        owner=owner,
                        repo=repo,
                        ecosystems=ecosystems_list,
                        tree_output=tree_output,
                        current_count=current_mined + successful_retries,
                        new_count=new_mined
                    )
                    successful_retries += 1
                else:
                    failed_retries += 1
                
                pbar.update(1)
        
        # Update output file header with new counts
        if successful_retries > 0:
            new_mined = current_mined + successful_retries
            update_output_file_header(
                output_file_path=str(output_file),
                old_mined_count=current_mined,
                new_mined_count=new_mined,
                total_count=total_packages
            )
            print(f"  ✓ Successfully retried: {successful_retries}")
            print(f"  ✗ Still failed: {failed_retries}")
            print(f"  📊 New total: {new_mined}/{total_packages} ({new_mined/total_packages*100:.1f}%)")
        else:
            print(f"  ✗ No successful retries")
        
        retry_summary[ecosystem_combination] = {
            'total_errors': len(errors),
            'successful_retries': successful_retries,
            'failed_retries': failed_retries,
            'old_mined': current_mined,
            'new_mined': current_mined + successful_retries,
            'total_packages': total_packages
        }
    
    # Write retry error logs for repositories that still failed
    print(f"\n{'=' * 80}")
    print("Writing retry error logs...")
    miner.error_log_dir = str(RETRY_ERROR_LOG_DIR)
    miner.write_error_logs()
    
    # Generate retry summary
    print(f"\n{'=' * 80}")
    print("Generating retry summary...")
    
    retry_summary_path = Path(output_dir) / "retry_summary.txt"
    
    with open(retry_summary_path, 'w', encoding='utf-8') as f:
        f.write("Retry Operation Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Ecosystem Combinations Processed: {len(retry_summary)}\n\n")
        
        total_attempted = sum(s['total_errors'] for s in retry_summary.values())
        total_successful = sum(s['successful_retries'] for s in retry_summary.values())
        total_failed = sum(s['failed_retries'] for s in retry_summary.values())
        
        f.write("=" * 80 + "\n")
        f.write("OVERALL RETRY STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Retry Attempts: {total_attempted}\n")
        f.write(f"Successful: {total_successful} ({total_successful/total_attempted*100:.2f}%)\n")
        f.write(f"Still Failed: {total_failed} ({total_failed/total_attempted*100:.2f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for combo, stats in sorted(retry_summary.items()):
            f.write(f"Ecosystem Combination: {combo}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Errors Attempted: {stats['total_errors']}\n")
            f.write(f"  Successful Retries: {stats['successful_retries']}\n")
            f.write(f"  Still Failed: {stats['failed_retries']}\n")
            f.write(f"  Previous Mined Count: {stats['old_mined']}/{stats['total_packages']}\n")
            f.write(f"  New Mined Count: {stats['new_mined']}/{stats['total_packages']}\n")
            
            old_rate = stats['old_mined'] / stats['total_packages'] * 100
            new_rate = stats['new_mined'] / stats['total_packages'] * 100
            improvement = new_rate - old_rate
            
            f.write(f"  Success Rate: {old_rate:.2f}% → {new_rate:.2f}% (+{improvement:.2f}%)\n")
            f.write("\n")
    
    print(f"✓ Retry summary saved to {retry_summary_path}")
    
    # Print final statistics
    print(f"\n{'=' * 80}")
    print("Retry Complete!")
    print(f"{'=' * 80}")
    print(f"Total Retry Attempts: {total_attempted}")
    print(f"Successful: {total_successful} ({total_successful/total_attempted*100:.1f}%)")
    print(f"Still Failed: {total_failed} ({total_failed/total_attempted*100:.1f}%)")
    print(f"{'=' * 80}\n")
    
    return retry_summary


def update_main_summary(output_dir: str, retry_summary: Dict):
    """
    Update the main summary.txt file with retry results.
    
    Args:
        output_dir: Directory containing the summary file
        retry_summary: Dictionary of retry results
    """
    summary_path = Path(output_dir) / "summary.txt"
    
    if not summary_path.exists():
        print(f"⚠ Summary file not found: {summary_path}")
        return
    
    print(f"\nUpdating main summary file: {summary_path}")
    
    # Read the current summary
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update statistics in the file
    for combo, stats in retry_summary.items():
        # Find and update the specific file entry
        pattern = rf"(File: {combo}\.csv\n.*?Mined Packages: )(\d+)\n(.*?Success Rate: )(\d+\.\d+)%"
        
        def replace_stats(match):
            old_mined = int(match.group(2))
            new_mined = stats['new_mined']
            new_rate = (new_mined / stats['total_packages']) * 100
            
            return f"{match.group(1)}{new_mined}\n{match.group(3)}{new_rate:.1f}%"
        
        content = re.sub(pattern, replace_stats, content, flags=re.DOTALL)
    
    # Update overall statistics
    # Parse current totals
    total_repos_match = re.search(r'Total Input Repositories: ([\d,]+)', content)
    success_match = re.search(r'Successfully Mined: ([\d,]+)', content)
    errors_match = re.search(r'Errors: ([\d,]+)', content)
    
    if total_repos_match and success_match and errors_match:
        total_repos = int(total_repos_match.group(1).replace(',', ''))
        old_success = int(success_match.group(1).replace(',', ''))
        old_errors = int(errors_match.group(1).replace(',', ''))
        
        # Calculate new totals
        total_successful_retries = sum(s['successful_retries'] for s in retry_summary.values())
        new_success = old_success + total_successful_retries
        new_errors = old_errors - total_successful_retries
        
        # Update overall statistics
        content = re.sub(
            r'Successfully Mined: [\d,]+ \(\d+\.\d+%\)',
            f'Successfully Mined: {new_success:,} ({new_success/total_repos*100:.2f}%)',
            content
        )
        content = re.sub(
            r'Errors: [\d,]+ \(\d+\.\d+%\)',
            f'Errors: {new_errors:,} ({new_errors/total_repos*100:.2f}%)',
            content
        )
        
        # Update error breakdown counts
        # This is more complex as we'd need to parse retry errors
        # For now, we'll add a note about the retry
        
        # Add retry information at the top
        retry_note = f"\n[UPDATED after retry on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
        retry_note += f"Retry added {total_successful_retries} previously failed repositories.\n"
        
        # Insert after the first line
        lines = content.split('\n')
        lines.insert(4, retry_note)
        content = '\n'.join(lines)
    
    # Write updated summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Summary file updated")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Retry failed repository mining and update output files"
    )
    parser.add_argument(
        "--error-log-dir",
        default=DEFAULT_ERROR_LOG_DIR,
        help=f"Directory containing error logs (default: {DEFAULT_ERROR_LOG_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--token",
        nargs="+",
        help="GitHub personal access token(s) (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum depth for directory structure (default: unlimited)",
    )
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="Update the main summary.txt file with retry results",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()
    error_log_dir = (script_dir / args.error_log_dir).resolve()
    
    # Create retry error log directory
    RETRY_ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 80}")
    print("Repository Retry Tool")
    print(f"{'=' * 80}")
    print(f"Error log directory: {error_log_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Retry error log directory: {RETRY_ERROR_LOG_DIR}")
    
    # Initialize miner
    miner = GitHubDirectoryMiner(
        github_tokens=args.token,
        error_log_dir=str(RETRY_ERROR_LOG_DIR)
    )
    
    # Prompt for tokens if not provided
    if not miner.tokens:
        print(f"\n{'=' * 80}")
        print("GitHub Token Required")
        print(f"{'=' * 80}")
        print("This script requires GitHub personal access token(s) to function.")
        print("You can:")
        print("  1. Set the GITHUB_TOKEN environment variable")
        print("  2. Pass token(s) via --token argument")
        print("  3. Enter token(s) now (separated by spaces)")
        print(f"{'=' * 80}\n")
        
        try:
            token_input = input("Enter GitHub token(s) [or press Enter to cancel]: ").strip()
            if token_input:
                miner.tokens = token_input.split()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
    
    if not miner.tokens:
        print("\n⚠ No GitHub tokens provided. Cannot proceed.")
        return
    
    # Validate tokens
    miner.validate_tokens()
    
    # Retry failed repositories
    retry_summary = retry_failed_repos(
        error_log_dir=str(error_log_dir),
        output_dir=str(output_dir),
        miner=miner,
        max_depth=args.max_depth
    )
    
    # Update main summary if requested
    if args.update_summary and retry_summary:
        update_main_summary(str(output_dir), retry_summary)


if __name__ == "__main__":
    main()
