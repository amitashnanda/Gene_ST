import os
import shutil
from pathlib import Path

def flatten_and_rename_images():
    source_dir = Path("/pscratch/sd/a/ananda/Spatial/Final_Code/dataset_raw/Whole_Slides_Segments")
    dest_dir = Path("/pscratch/sd/a/ananda/Spatial/Final_Code/dataset_raw/Whole_Slides_Segments_Flattened")
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    
    copied_count = 0
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.JPG')):
                file_path = Path(root) / file
                
                rel_path = file_path.parent.relative_to(source_dir)
                parts = rel_path.parts
                
                
                stem = file_path.stem 
                suffix = file_path.suffix 
                
                if len(parts) >= 2:
                    
                    grandparent = parts[0]
                    parent = parts[-1] 
                    
                    new_filename = f"{stem}_{parent}_{grandparent}{suffix}"
                elif len(parts) == 1:
                   
                    new_filename = f"{stem}_{parts[0]}{suffix}"
                else:
                   
                    new_filename = file
                
                dest_path = dest_dir / new_filename
                
               
                if dest_path.exists():
                    print(f"Warning: {new_filename} already exists. Skipping or renaming...")
                    
                    counter = 1
                    while dest_path.exists():
                        new_filename = f"{stem}_{parent}_{grandparent}_{counter}{suffix}"
                        dest_path = dest_dir / new_filename
                        counter += 1
                
                shutil.copy2(file_path, dest_path)
                copied_count += 1
                if copied_count % 100 == 0:
                    print(f"Copied {copied_count} images...")

    print(f"Finished! Copied {copied_count} images to {dest_dir}")

if __name__ == "__main__":
    flatten_and_rename_images()
