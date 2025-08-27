"""
GeneST_patch_selector.py
----------------------
Interactive patch selector for GeneST dataset with zoom functionality.
This script allows users to select patches from a whole slide image (WSI) interactively,
Zoom in/out, and save the selected patches along with their coordinates.
It uses OpenCV for GUI and image processing, and PIL for image handling.

"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  

def select_patches(image_path, patch_dir, coords_txt):
    """
    Select patches from a whole slide image (WSI) interactively.
    Args:
        image_path (str): Path to the input image (TIFF/PNG/JPG).
        patch_dir (str): Directory to save the selected patches.
        coords_txt (str): File to save the coordinates of selected patches.

    This function allows the user to zoom in/out on the image, draw rectangles to select patches, and save the selected patches along with their coordinates
    in a text file. The user can interactively select patches by clicking and dragging the mouse over the image. The selected patches are saved as PNG files in the specified
    directory, and their coordinates are logged in a text file.
    
    The user can exit the selection mode by pressing 'q'.

    The patches are saved in the format:
        patch_name xmin ymin xmax ymax width height
    where:
        - patch_name: Name of the saved patch file (e.g., patch_0001.png)
        - xmin, ymin: Top-left corner coordinates of the patch
        - xmax, ymax: Bottom-right corner coordinates of the patch
        - width, height: Width and height of the patch

    Example usage:
        python GeneST_patch_selector.py path/to/image.png --patch_dir patches_raw --coords_txt patch_coords.txt
    This will open the image in a window, allow the user to select patches interactively,
    and save the selected patches in the 'patches_raw' directory with their coordinates
    saved in 'patch_coords.txt'.    

    """

    os.makedirs(patch_dir, exist_ok=True)

    pil_img  = Image.open(image_path).convert("RGB")
    img_rgb  = np.array(pil_img)                     
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


    win = "GeneST"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    current_scale = 1.0
    disp_img      = img_bgr.copy()

    def redraw(scale):
        nonlocal disp_img, current_scale
        current_scale = scale
        h, w = img_bgr.shape[:2]
        disp_img = cv2.resize(img_bgr, (int(w*scale), int(h*scale)),
                              interpolation=cv2.INTER_AREA)
        cv2.imshow(win, disp_img)
    def on_trackbar(v):
        v = max(v, 1)             
        redraw(scale=v/10.0)

    cv2.createTrackbar("Zoom", win, 10, 80, on_trackbar)  

    redraw(1.0)  
    drawing, ix, iy, count = False, 0, 0, 0
    records = []

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, ix, iy, count
        view = disp_img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing, ix, iy = True, x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.rectangle(view, (ix, iy), (x, y), (0,255,0), 2)
            cv2.imshow(win, view)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            s = current_scale
            x0, y0 = int(min(ix,x)/s), int(min(iy,y)/s)
            x1, y1 = int(max(ix,x)/s), int(max(iy,y)/s)
            w, h   = x1-x0, y1-y0
            if w == 0 or h == 0:
                return
            count += 1
            name  = f"patch_{count:04d}.png"
            path  = os.path.join(patch_dir, name)
            patch = img_rgb[y0:y1, x0:x1]           
            Image.fromarray(patch).save(path, format="PNG")
            print(f"[{count:04d}] {name}: ({x0},{y0})→({x1},{y1})  {w}×{h}")
            records.append((name, x0, y0, x1, y1, w, h))
            cv2.imshow(win, disp_img)               

    cv2.setMouseCallback(win, mouse_cb)
    print("GeneST")

    while True:
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    with open(coords_txt, "w") as f:
        f.write("patch_name xmin ymin xmax ymax width height\n")
        for rec in records:
            f.write(" ".join(map(str, rec)) + "\n")
    print("Coordinates saved to", coords_txt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Interactive patch selector with zoom")
    ap.add_argument("image",       help="Path to large WSI (TIFF/PNG/JPG)")
    ap.add_argument("--patch_dir", default="patches_raw",
                    help="Directory to save patches")
    ap.add_argument("--coords_txt",default="patch_coords.txt",
                    help="File to save patch coordinates")
    args = ap.parse_args()
    select_patches(args.image, args.patch_dir, args.coords_txt)
