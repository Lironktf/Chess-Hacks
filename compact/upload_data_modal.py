"""
Upload training data to Modal storage

This uploads your PGN files to Modal so they can be used for training
"""

import modal

app = modal.App("chess-data-upload")

# Create volume for data
data_volume = modal.Volume.from_name("chess-data", create_if_missing=True)

# Simple image
image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=3600,  # 1 hour timeout for upload
)
def upload_files(file_contents_dict):
    """Upload PGN files to Modal volume"""
    import os

    print("Uploading files to Modal storage...")
    print()

    for filename, content in file_contents_dict.items():
        filepath = f"/data/{filename}"
        print(f"  Uploading {filename} ({len(content) / 1024 / 1024:.1f} MB)...")

        with open(filepath, 'wb') as f:
            f.write(content)

        print(f"  ✓ Saved to {filepath}")

    print()
    print(f"✓ Uploaded {len(file_contents_dict)} files")

    # List files in volume
    print()
    print("Files in Modal storage:")
    for f in os.listdir("/data"):
        size = os.path.getsize(f"/data/{f}") / 1024 / 1024
        print(f"  {f}: {size:.1f} MB")

    # Commit changes
    data_volume.commit()
    print()
    print("✓ Volume committed")


@app.local_entrypoint()
def main():
    """Upload local PGN files to Modal"""
    import os

    print("=" * 60)
    print("UPLOAD CHESS DATA TO MODAL")
    print("=" * 60)
    print()

    data_dir = "../data"

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    # Find PGN files
    pgn_files = [f for f in os.listdir(data_dir) if f.endswith(".pgn")]

    if not pgn_files:
        print(f"Error: No PGN files found in {data_dir}")
        return

    print(f"Found {len(pgn_files)} PGN files:")
    print()

    total_size = 0
    for f in pgn_files:
        size = os.path.getsize(os.path.join(data_dir, f)) / 1024 / 1024
        total_size += size
        print(f"  {f}: {size:.1f} MB")

    print()
    print(f"Total size: {total_size:.1f} MB")
    print()

    # Modal free tier has limits, warn if too large
    if total_size > 5000:  # 5 GB
        print("⚠️  WARNING: Data is very large (> 5 GB)")
        print("   Consider uploading fewer files or filtering first")
        print()

    confirm = input("Upload these files to Modal? (y/n): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        return

    print()
    print("Reading files...")
    file_contents = {}

    for filename in pgn_files:
        filepath = os.path.join(data_dir, filename)
        print(f"  Reading {filename}...")
        with open(filepath, 'rb') as f:
            file_contents[filename] = f.read()

    print()
    print("Uploading to Modal (this may take a few minutes)...")
    print()

    # Upload to Modal
    upload_files.remote(file_contents)

    print()
    print("=" * 60)
    print("UPLOAD COMPLETE!")
    print("=" * 60)
    print()
    print("Your data is now in Modal storage.")
    print()
    print("Next step:")
    print("  modal run train_modal.py")
    print()
