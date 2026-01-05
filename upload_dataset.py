"""
Upload g-images-dataset (Xu 2014) to Modal volume.

Usage:
    python -m modal run upload_xu2014_to_modal.py
    python -m modal run upload_xu2014_to_modal.py --verify-only
    python -m modal run upload_xu2014_to_modal.py --clear-first
"""

import modal
from pathlib import Path
import os

APP_NAME = "archstyle-upload-xu2014"
DATA_VOL_NAME = "archstyle-data-xu2014"
LOCAL_DATA_DIR = Path("g-images-dataset")

EXCLUDE_DIRS = {"__pycache__", ".git"}

app = modal.App(name=APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=60 * 60,
)
def clear_volume():
    """Clear all data from the volume."""
    import shutil
    from pathlib import Path
    
    data_dir = Path("/data")
    cleared = 0
    
    for item in list(data_dir.iterdir()):
        if item.is_dir():
            count = len(list(item.rglob("*")))
            shutil.rmtree(item)
            cleared += count
            print(f"  Removed: {item.name}")
    
    data_volume.commit()
    print(f"Cleared {cleared} items from volume")
    return cleared


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=60 * 60,
)
def upload_class(class_name: str, images: list):
    """Upload images for a single class."""
    from pathlib import Path
    
    cls_dir = Path("/data") / class_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    
    existing = set(f.name for f in cls_dir.iterdir() if f.is_file())
    
    uploaded = 0
    skipped = 0
    for filename, img_bytes in images:
        if filename in existing:
            skipped += 1
            continue
        dest_path = cls_dir / filename
        with open(dest_path, 'wb') as f:
            f.write(img_bytes)
        existing.add(filename)
        uploaded += 1
    
    data_volume.commit()
    return {"uploaded": uploaded, "skipped": skipped}


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=60 * 60,
)
def verify_dataset():
    """Verify dataset contents."""
    from pathlib import Path
    
    data_dir = Path("/data")
    
    print("\n" + "=" * 60)
    print("XU2014 DATASET VERIFICATION")
    print("=" * 60)
    
    total = 0
    class_counts = {}
    
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            files = list(item.glob("*.jpg")) + list(item.glob("*.JPG")) + list(item.glob("*.png"))
            count = len(files)
            if count > 0:
                class_counts[item.name] = count
                total += count
    
    print("\nClass counts:")
    for name, count in sorted(class_counts.items()):
        print(f"  {name}: {count}")
    
    print(f"\nTotal classes: {len(class_counts)}")
    print(f"Total images: {total}")
    
    return {"total": total, "classes": len(class_counts), "class_counts": class_counts}


@app.local_entrypoint()
def main(verify_only: bool = False, clear_first: bool = False):
    """Upload g-images-dataset to Modal."""
    
    if verify_only:
        print("Verifying Modal xu2014 dataset...")
        result = verify_dataset.remote()
        return
    
    print("=" * 60)
    print("UPLOADING XU2014 DATASET TO MODAL")
    print("=" * 60)
    
    if clear_first:
        print("\nClearing existing data...")
        clear_volume.remote()
    
    # Find all class directories locally
    classes = []
    for d in sorted(LOCAL_DATA_DIR.iterdir()):
        if d.is_dir() and d.name not in EXCLUDE_DIRS:
            images = []
            seen = set()
            try:
                for filename in os.listdir(str(d)):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        if filename not in seen:
                            seen.add(filename)
                            images.append(d / filename)
            except Exception as e:
                print(f"  Warning: Error reading {d.name}: {e}")
            if images:
                classes.append((d.name, images))
    
    print(f"\nFound {len(classes)} classes locally:")
    total_local = 0
    for name, imgs in classes:
        print(f"  {name}: {len(imgs)}")
        total_local += len(imgs)
    print(f"Total unique images: {total_local}")
    
    # Upload each class
    print("\nUploading to Modal...")
    total_uploaded = 0
    total_skipped = 0
    total_errors = 0
    
    for class_name, img_paths in classes:
        images = []
        errors = 0
        for img_path in img_paths:
            try:
                with open(str(img_path), 'rb') as f:
                    img_data = f.read()
                
                try:
                    safe_name = img_path.name.encode('ascii', 'ignore').decode('ascii')
                except:
                    safe_name = ""
                
                if not safe_name or len(safe_name) < 5:
                    ext = img_path.suffix if img_path.suffix else '.jpg'
                    safe_name = f"img_{len(images):05d}{ext}"
                
                images.append((safe_name, img_data))
            except Exception as e:
                errors += 1
        
        if errors > 0:
            total_errors += errors
        
        result = upload_class.remote(class_name, images)
        total_uploaded += result["uploaded"]
        total_skipped += result["skipped"]
        status = f"{result['uploaded']} uploaded, {result['skipped']} skipped"
        if errors > 0:
            status += f", {errors} read errors"
        print(f"  {class_name}: {status}")
    
    # Verify
    print("\nVerifying upload...")
    result = verify_dataset.remote()
    
    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print(f"  Local unique images: {total_local}")
    print(f"  Uploaded: {total_uploaded}")
    print(f"  Skipped (already existed): {total_skipped}")
    print(f"  Modal total: {result['total']}")
    if result['total'] == total_local:
        print("  [OK] Counts match!")
    else:
        print(f"  [WARNING] Mismatch! Diff: {result['total'] - total_local}")
    print("=" * 60)

