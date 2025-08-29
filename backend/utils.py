import os
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def find_images(root):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                out.append(os.path.join(dirpath, fn))
    return out

def rel_to(path, root):
    # Make path relative to root
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def thumb_from_face(img: Image.Image, bbox, size=160, pad=0.2):
    x1, y1, x2, y2 = bbox
    w, h = img.size
    # expand bbox by pad
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    s = int(max(bw, bh) * (1 + pad))
    nx1 = max(0, int(cx - s/2))
    ny1 = max(0, int(cy - s/2))
    nx2 = min(w, int(cx + s/2))
    ny2 = min(h, int(cy + s/2))
    crop = img.crop((nx1, ny1, nx2, ny2))
    crop.thumbnail((size, size))
    return crop