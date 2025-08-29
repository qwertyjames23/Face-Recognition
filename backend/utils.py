import os
from PIL import Image, ImageOps

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
    # Handle EXIF orientation: if present, map bbox coordinates then transpose image
    try:
        exif = img.getexif()
        orientation = exif.get(274)
    except Exception:
        orientation = None

    w, h = img.size

    def map_coord(x, y, ori, w, h):
        # Map a single coordinate depending on EXIF orientation
        if not ori or ori == 1:
            return x, y
        if ori == 2:
            return w - x, y
        if ori == 3:
            return w - x, h - y
        if ori == 4:
            return x, h - y
        if ori == 5:
            return y, x
        if ori == 6:
            return y, w - x
        if ori == 7:
            return h - y, w - x
        if ori == 8:
            return h - y, x
        return x, y

    # transform bbox corner points if orientation is present
    try:
        x1, y1, x2, y2 = bbox
        if orientation and orientation != 1:
            pts = [map_coord(x1, y1, orientation, w, h),
                   map_coord(x2, y2, orientation, w, h),
                   map_coord(x1, y2, orientation, w, h),
                   map_coord(x2, y1, orientation, w, h)]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1t, x2t = int(min(xs)), int(max(xs))
            y1t, y2t = int(min(ys)), int(max(ys))
            x1, y1, x2, y2 = x1t, y1t, x2t, y2t
    except Exception:
        # fallback: use provided bbox
        x1, y1, x2, y2 = bbox

    # transpose the image so it's upright for display
    try:
        img_t = ImageOps.exif_transpose(img)
    except Exception:
        img_t = img

    # After transposing, image size may have changed
    w2, h2 = img_t.size

    # expand bbox by pad around the center of the bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    s = int(max(bw, bh) * (1 + pad))
    nx1 = max(0, int(cx - s/2))
    ny1 = max(0, int(cy - s/2))
    nx2 = min(w2, int(cx + s/2))
    ny2 = min(h2, int(cy + s/2))
    crop = img_t.crop((nx1, ny1, nx2, ny2))
    crop.thumbnail((size, size))
    return crop