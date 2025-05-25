
#!/usr/bin/env python3
import os
import argparse
from datetime import datetime
from pathlib import Path

def get_creation_time(file_path):
    """Get the creation time of a file and format it as YYYY-MM-DD_HH-MM-SS"""
    try:
        # Get creation time (or modification time if creation time is not available)
        if hasattr(os.stat(file_path), 'st_birthtime'):
            # macOS
            creation_time = os.stat(file_path).st_birthtime
        else:
            # Windows/Linux - use the earliest of creation or modification time
            stat = os.stat(file_path)
            creation_time = min(stat.st_ctime, stat.st_mtime)

        # Convert to datetime and format
        dt = datetime.fromtimestamp(creation_time)
        return dt.strftime("%Y-%m-%d_%H-%M-%S")
    except Exception as e:
        print(f"Error getting creation time for {file_path}: {e}")
        return None

def rename_files_in_directory(input_dir, dry_run=False):
    """Rename all files in subdirectories to their creation date-time"""
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist")
        return

    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return

    renamed_count = 0
    error_count = 0

    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            file_path = Path(root) / filename

            # Skip if it's not a file
            if not file_path.is_file():
                continue

            # Get creation time
            creation_time_str = get_creation_time(file_path)
            if not creation_time_str:
                error_count += 1
                continue

            # Get file extension
            file_extension = file_path.suffix

            # Create new filename
            new_filename = f"{creation_time_str}{file_extension}"
            new_file_path = file_path.parent / new_filename

            # Handle duplicate names by adding a counter
            counter = 1
            original_new_path = new_file_path
            while new_file_path.exists() and new_file_path != file_path:
                name_without_ext = f"{creation_time_str}_{counter:03d}"
                new_file_path = file_path.parent / f"{name_without_ext}{file_extension}"
                counter += 1

            # Skip if the file already has the correct name
            if new_file_path == file_path:
                print(f"Skipping: {file_path} (already has correct name)")
                continue

            try:
                if dry_run:
                    print(f"Would rename: {file_path} -> {new_file_path}")
                else:
                    file_path.rename(new_file_path)
                    print(f"Renamed: {file_path} -> {new_file_path}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {file_path}: {e}")
                error_count += 1

    print(f"\nSummary:")
    print(f"Files {'would be ' if dry_run else ''}renamed: {renamed_count}")
    print(f"Errors: {error_count}")

def main():
    parser = argparse.ArgumentParser(
        description="Rename files in subdirectories to their creation date-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rename_files.py --input_dir ./photos
  python rename_files.py --input_dir /path/to/files --dry-run
        """
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing subdirectories with files to rename'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be renamed without actually renaming files'
    )

    args = parser.parse_args()

    print(f"Processing files in: {args.input_dir}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be actually renamed")
    print("-" * 50)

    rename_files_in_directory(args.input_dir, args.dry_run)

if __name__ == "__main__":
    main()
