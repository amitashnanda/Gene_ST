import os
import sys
import argparse
import numpy as np

from PIL import Image, ImageOps

# Optional import (only needed for .czi)
try:
    import aicspylibczi
except ImportError:
    aicspylibczi = None

from PyQt5.QtWidgets import QApplication, QMainWindow, QStatusBar, QRubberBand, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QRectF, QPointF, QSizeF

Image.MAX_IMAGE_PIXELS = None

# --------------------------
# Utils
# --------------------------

def numpy_to_qpixmap(np_img_bgr):
    """
    Converts a NumPy array (H x W x 3, assumed BGR uint8) to a QPixmap.
    If grayscale (H x W), it will be stacked to 3 channels first.
    """
    if np_img_bgr is None:
        return QPixmap()
    if np_img_bgr.ndim == 2:
        np_img_bgr = np.stack((np_img_bgr,) * 3, axis=-1)

    # Convert BGR->RGB for Qt
    np_rgb = np_img_bgr[:, :, ::-1].copy()
    h, w, ch = np_rgb.shape
    q_img = QImage(np_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)


# --------------------------
# Image Backends
# --------------------------

class ImageBackendBase:
    """Abstract interface for image sources."""
    def get_size(self):
        raise NotImplementedError

    def get_offsets(self):
        """(offset_x, offset_y) for absolute coords; CZI uses scene bbox offset, others (0,0)."""
        return (0, 0)

    def get_thumbnail_bgr(self, display_downsample_factor):
        """Return HxWx3 uint8 BGR thumbnail for display."""
        raise NotImplementedError

    def read_region_rgb(self, x, y, w, h):
        """Return HxWx3 uint8 RGB patch from the full-resolution image."""
        raise NotImplementedError


class CZIBackend(ImageBackendBase):
    def __init__(self, path):
        if aicspylibczi is None:
            raise RuntimeError("aicspylibczi is not installed but a .czi file was provided.")
        self.czi = aicspylibczi.CziFile(path)
        self.bbox = self.czi.get_mosaic_scene_bounding_box()  # has x,y,w,h
        self._offset_x, self._offset_y = self.bbox.x, self.bbox.y
        self._w, self._h = self.bbox.w, self.bbox.h

        dims_size = self.czi.size
        if 'C' in dims_size and dims_size['C'][0] >= 3:
            self.channels_to_read = [0, 1, 2]
        else:
            # fallback: single channel (grayscale)
            self.channels_to_read = 0

    def get_size(self):
        return (self._w, self._h)

    def get_offsets(self):
        return (self._offset_x, self._offset_y)

    def _to_hwc(self, arr):
        """CZI returns (1, C, H, W) or similar; normalize to HxWxC."""
        arr = np.squeeze(arr[0])
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            # channel-first -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    def get_thumbnail_bgr(self, display_downsample_factor):
        scale = 1.0 / float(display_downsample_factor)
        thumb = self.czi.read_mosaic(C=self.channels_to_read, scale_factor=scale)
        thumb = self._to_hwc(thumb)

        if thumb.ndim == 2:
            thumb = np.stack([thumb, thumb, thumb], axis=-1)

        # The original pipeline assumed CZI data is BGR for display -> keep that contract.
        # If your CZI data is actually RGB, flip here to BGR:
        # thumb = thumb[:, :, ::-1]
        # However, to match your earlier code path, leave as-is and treat as BGR downstream.
        # To be safe, convert to uint8 if not already:
        if thumb.dtype != np.uint8:
            # naive rescale for visualization if needed
            thumb = np.clip(thumb, 0, 255).astype(np.uint8)
        return thumb

    def read_region_rgb(self, x, y, w, h):
        patch = self.czi.read_mosaic(region=(int(x), int(y), int(w), int(h)),
                                     scale_factor=1.0,
                                     C=self.channels_to_read)
        patch = self._to_hwc(patch)

        if patch.ndim == 2:
            patch = np.stack([patch, patch, patch], axis=-1)

        # Your original code saved patches after converting BGR->RGB.
        # Do the same here to return RGB for saving.
        if patch.dtype != np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)
        patch_rgb = patch[:, :, ::-1]  # BGR -> RGB
        return patch_rgb


class PILBackend(ImageBackendBase):
    def __init__(self, path):
        self.img = Image.open(path)
        # Fix orientation via EXIF if present
        self.img = ImageOps.exif_transpose(self.img)
        # Convert to a canonical mode for region reads; keep original for patch extraction
        # We won't convert here; we'll convert as needed.
        self._w, self._h = self.img.size

    def get_size(self):
        return (self._w, self._h)

    def get_thumbnail_bgr(self, display_downsample_factor):
        new_w = max(1, int(self._w / float(display_downsample_factor)))
        new_h = max(1, int(self._h / float(display_downsample_factor)))
        thumb = self.img.resize((new_w, new_h), resample=Image.BILINEAR)

        # For display, produce BGR uint8
        if thumb.mode not in ("RGB", "RGBA", "L"):
            thumb = thumb.convert("RGB")
        if thumb.mode == "RGBA":
            thumb = thumb.convert("RGB")
        if thumb.mode == "L":
            thumb = Image.merge("RGB", (thumb, thumb, thumb))

        thumb_np = np.array(thumb)  # RGB uint8
        bgr = thumb_np[:, :, ::-1]  # -> BGR
        return bgr

    def read_region_rgb(self, x, y, w, h):
        box = (int(x), int(y), int(x + w), int(y + h))
        patch = self.img.crop(box)
        if patch.mode not in ("RGB", "RGBA", "L"):
            patch = patch.convert("RGB")
        if patch.mode == "RGBA":
            patch = patch.convert("RGB")
        if patch.mode == "L":
            patch = Image.merge("RGB", (patch, patch, patch))
        patch_np = np.array(patch)  # RGB uint8
        return patch_np


def make_backend(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".czi":
        return CZIBackend(image_path)
    elif ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".jpg"):
        return PILBackend(image_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .czi, .tif/.tiff, .jpg/.jpeg, or .png.")


# --------------------------
# Viewer Widgets
# --------------------------

class PhotoViewer(QWidget):
    """A custom widget to display and interact with a QPixmap, preserving aspect ratio."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap()
        self.target_rect = QRectF()
        self.setStyleSheet("background-color: #404040;")

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        self.target_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        self.update()

    def get_display_rect(self):
        """Calculates the aspect-ratio-correct rectangle to draw the image in."""
        if self.pixmap.isNull() or not self.target_rect.isValid() or self.target_rect.height() == 0:
            return QRectF()

        widget_rect = self.rect()
        source_rect = self.target_rect

        widget_ar = widget_rect.width() / widget_rect.height()
        source_ar = source_rect.width() / source_rect.height()

        dest_rect = QRectF()
        if widget_ar > source_ar:
            new_height = widget_rect.height()
            new_width = new_height * source_ar
            dest_rect.setSize(QSizeF(new_width, new_height))
        else:
            new_width = widget_rect.width()
            new_height = new_width / source_ar
            dest_rect.setSize(QSizeF(new_width, new_height))

        dest_rect.moveCenter(widget_rect.center())
        return dest_rect

    def paintEvent(self, event):
        if self.pixmap.isNull():
            return

        # Note: QPainter context manager support varies by PyQt versions; keep as-is per user code.
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            destination_rect = self.get_display_rect()
            painter.drawPixmap(destination_rect, self.pixmap, self.target_rect)

    def map_widget_to_pixmap(self, widget_pos):
        if self.pixmap.isNull() or not self.target_rect.isValid():
            return QPointF()

        pixmap_display_rect = self.get_display_rect()
        if not pixmap_display_rect.contains(widget_pos):
            return QPointF()

        x_prop = (widget_pos.x() - pixmap_display_rect.left()) / pixmap_display_rect.width()
        y_prop = (widget_pos.y() - pixmap_display_rect.top()) / pixmap_display_rect.height()

        target_x = self.target_rect.left() + self.target_rect.width() * x_prop
        target_y = self.target_rect.top() + self.target_rect.height() * y_prop
        return QPointF(target_x, target_y)


# --------------------------
# Main Window
# --------------------------

class ImageSelector(QMainWindow):
    def __init__(self, image_path, patch_dir, coords_txt):
        super().__init__()
        self.image_path = image_path
        self.patch_dir = patch_dir
        self.coords_txt = coords_txt

        self.backend = None
        self.offset_x, self.offset_y = 0, 0
        self.display_downsample_factor = 1.0
        self.records = []
        self.patch_count = 0
        self.mode = 'navigate'

        self.viewer = PhotoViewer(self)
        self.setCentralWidget(self.viewer)
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewer)

        self.setWindowTitle("GeneST Image Patcher (CZI / TIFF / JPEG)")
        self.resize(1200, 800)
        self.setup_statusbar()
        self.load_image_and_thumbnail()

    def setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_statusbar()

    def update_statusbar(self):
        if self.mode == 'navigate':
            message = "[NAVIGATE MODE] Mouse Wheel: Zoom | Click+Drag: Pan | Press 's' for Select Mode"
        else:
            message = "[SELECT MODE] Click+Drag: Draw box to save patch | Press 'n' for Navigate Mode"
        self.status_bar.showMessage(message)

    def load_image_and_thumbnail(self):
        print("Loading image and creating thumbnail...")
        self.backend = make_backend(self.image_path)

        full_width, full_height = self.backend.get_size()
        self.offset_x, self.offset_y = self.backend.get_offsets()

        max_display_dim = 5000.0
        # Downsample factor = how many original pixels per displayed pixel
        self.display_downsample_factor = max(full_width, full_height) / max_display_dim

        # Thumbnail for display (BGR for numpy_to_qpixmap)
        thumb_bgr = self.backend.get_thumbnail_bgr(self.display_downsample_factor)
        thumbnail_pixmap = numpy_to_qpixmap(thumb_bgr)
        self.viewer.setPixmap(thumbnail_pixmap)
        print(f"Thumbnail ready. Size: {full_width} x {full_height}")

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_N:
            self.mode = 'navigate'
            self.update_statusbar()
        elif key == Qt.Key_S:
            self.mode = 'select'
            self.update_statusbar()
        elif key == Qt.Key_Q:
            self.close()

    def wheelEvent(self, event):
        if self.mode == 'navigate':
            anchor_point = self.viewer.map_widget_to_pixmap(event.pos())
            if anchor_point.isNull():
                return

            steps = event.angleDelta().y() / 120
            factor = 1.15 ** (-steps)

            new_width = self.viewer.target_rect.width() * factor
            new_height = self.viewer.target_rect.height() * factor

            new_x = anchor_point.x() - (anchor_point.x() - self.viewer.target_rect.left()) * factor
            new_y = anchor_point.y() - (anchor_point.y() - self.viewer.target_rect.top()) * factor

            self.viewer.target_rect = QRectF(new_x, new_y, new_width, new_height)
            self.viewer.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pan_start_pos = event.pos()
            if self.mode == 'select':
                self.rubber_band.setGeometry(QRect(self.pan_start_pos, QSize()))
                self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            if self.mode == 'navigate':
                if self.viewer.width() == 0 or self.viewer.height() == 0:
                    return
                delta = event.pos() - self.pan_start_pos
                pixmap_delta_x = delta.x() * (self.viewer.target_rect.width() / self.viewer.width())
                pixmap_delta_y = delta.y() * (self.viewer.target_rect.height() / self.viewer.height())

                self.viewer.target_rect.translate(-pixmap_delta_x, -pixmap_delta_y)
                self.pan_start_pos = event.pos()
                self.viewer.update()
            elif self.mode == 'select':
                self.rubber_band.setGeometry(QRect(self.pan_start_pos, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.mode == 'select':
            self.rubber_band.hide()
            selection_rect_widget = self.rubber_band.geometry()

            top_left = self.viewer.map_widget_to_pixmap(selection_rect_widget.topLeft())
            bottom_right = self.viewer.map_widget_to_pixmap(selection_rect_widget.bottomRight())

            if top_left.isNull() or bottom_right.isNull():
                return

            self.process_selection(QRectF(top_left, bottom_right))

    def closeEvent(self, event):
        print("\nSaving coordinates file...")
        os.makedirs(os.path.dirname(self.coords_txt) or ".", exist_ok=True)
        with open(self.coords_txt, "w") as f:
            f.write("patch_name xmin ymin xmax ymax width height\n")
            for rec in self.records:
                f.write(" ".join(map(str, rec)) + "\n")
        print(f"Coordinates for {self.patch_count} patches saved to {self.coords_txt}")
        event.accept()

    def process_selection(self, thumb_rect):
        # Thumb coordinates (in target_rect pixel space)
        x0_on_thumb = thumb_rect.left()
        y0_on_thumb = thumb_rect.top()
        x1_on_thumb = thumb_rect.right()
        y1_on_thumb = thumb_rect.bottom()

        # Map to absolute coordinates of full-resolution image (account for mosaic offsets, if any)
        x0_abs = int(x0_on_thumb * self.display_downsample_factor) + self.offset_x
        y0_abs = int(y0_on_thumb * self.display_downsample_factor) + self.offset_y
        x1_abs = int(x1_on_thumb * self.display_downsample_factor) + self.offset_x
        y1_abs = int(y1_on_thumb * self.display_downsample_factor) + self.offset_y

        w_orig, h_orig = x1_abs - x0_abs, y1_abs - y0_abs
        if w_orig <= 0 or h_orig <= 0:
            return

        # Ensure output dir exists
        os.makedirs(self.patch_dir, exist_ok=True)

        self.patch_count += 1
        name = f"patch_{self.patch_count:04d}.png"
        path = os.path.join(self.patch_dir, name)

        print(f"Extracting high-res patch [{self.patch_count:04d}]... ({w_orig}x{h_orig})")
        patch_rgb = self.backend.read_region_rgb(x0_abs, y0_abs, w_orig, h_orig)  # HxWx3 RGB uint8

        if patch_rgb is not None and patch_rgb.size > 0:
            Image.fromarray(patch_rgb).save(path, format="PNG")
            print(f"  > Saved {name}")
            self.records.append((name, x0_abs, y0_abs, x1_abs, y1_abs, w_orig, h_orig))


# --------------------------
# CLI
# --------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Interactive patch selector (CZI / TIFF / JPEG / PNG) using PyQt5")
    ap.add_argument("image", help="Path to image file (.czi, .tif/.tiff, .jpg/.jpeg, .png)")
    ap.add_argument("--patch_dir", default="patches_raw", help="Directory to save patches")
    ap.add_argument("--coords_txt", default="patch_coords.txt", help="File to save patch coordinates")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    main_win = ImageSelector(args.image, args.patch_dir, args.coords_txt)
    main_win.show()
    sys.exit(app.exec_())
