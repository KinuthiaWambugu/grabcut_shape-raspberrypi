from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import time

def detect_four_circle_points(img_rgb: np.ndarray,
                              min_radius_frac: float = 0.01,
                              max_radius_frac: float = 0.12,
                              padding_frac: float = 0.02,
                              debug: bool = False
                              ) -> tuple[tuple[int,int,int,int], np.ndarray]:

    h, w = img_rgb.shape[:2]
    min_dim = min(h, w)
    min_r = max(1, int(min_dim * min_radius_frac))
    max_r = int(min_dim * max_radius_frac)
    pad_x = int(w * padding_frac)
    pad_y = int(h * padding_frac)

    t0 = time.perf_counter()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    print(f"Blur: {(time.perf_counter()-t0)*1000:.1f}ms")

    centres = None

    # Pass 1: Hough circles
    t0 = time.perf_counter()
    raw = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,  #original = 1.2
        minDist=min_dim * 0.12,  #separation=0.08 
        param1=100,               #original edges = 50
        param2=35,                #Circle threshold = 25
        minRadius=min_r,
        maxRadius=max_r,
    )
    print(f"Hough: {(time.perf_counter()-t0)*1000:.1f}ms")
    if raw is not None:
        raw = np.round(raw[0]).astype(int)
        if debug:
            print(f"  Hough found {len(raw)} circles")
        if len(raw) >= 4:
            centres = raw[:, :2]

    # ── Pass 2: contour-based fallback ───────────────────────────────────
    if centres is None or len(centres) < 4:
        if debug:
            print("  Hough insufficient – falling back to contour detection")
        _, thresh = cv2.threshold(
            cv2.equalizeHist(gray), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        blob_pts = []
        min_area = np.pi * min_r ** 2
        max_area = np.pi * max_r ** 2
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            blob_pts.append((cx, cy, area))
        # Sort by area desc, take top candidates, remove duplicates within min_dist
        blob_pts.sort(key=lambda t: -t[2])
        filtered = []
        for bx, by, _ in blob_pts:
            if all(np.hypot(bx - fx, by - fy) > min_dim * 0.06
                   for fx, fy in filtered):
                filtered.append((bx, by))
        if debug:
            print(f"  Contour fallback: {len(filtered)} candidate blobs")
        if len(filtered) >= 4:
            centres = np.array(filtered[:20], dtype=int)

    # ── Select the 4 circles that maximise the convex hull area 
    if centres is not None and len(centres) >= 4:
        if len(centres) == 4:
            if debug:
                print(f"Found exactly 4 circles")
        else:
            from itertools import combinations
            best_area = -1
            best_quad = None
            # Cap exhaustive search at C(min(20,N), 4) combos
            t0=time.perf_counter()
            search_pts = centres[:20]
            for quad in combinations(range(len(search_pts)), 4):
                pts4 = search_pts[list(quad)]
                hull = cv2.convexHull(pts4.reshape(-1, 1, 2).astype(np.float32))
                area = cv2.contourArea(hull)
                if area > best_area:
                    best_area = area
                    best_quad = pts4
            centres = best_quad  # (4, 2)
            print(f"Quad search: {(time.perf_counter()-t0)*1000:.1f}ms")
    else:
        raise ValueError(
            f"Could not detect 4 circles in this image. "
            f"Found {0 if centres is None else len(centres)} candidate(s). "
            f"Consider adjusting min/max_radius_frac or using center_crop fallback."
        )

    # ── Build padded bounding box ────────────────────────────────────────
    xs, ys = centres[:, 0], centres[:, 1]
    x_min = max(0, xs.min() - pad_x)
    y_min = max(0, ys.min() - pad_y)
    x_max = min(w - 1, xs.max() + pad_x)
    y_max = min(h - 1, ys.max() + pad_y)

    bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    return bbox, centres

def load_and_preprocess_image(
        image_path: Path | str,
        target_size: Tuple[int, int] = (480, 360),
        color_space: str = "rgb",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load, resize and convert color space."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at path: {path.resolve()}")
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"OpenCV failed to decode image: {path.resolve()}")

    resized_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    cs = color_space.lower()
    if cs == "rgb":
        converted = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    elif cs == "hsv":
        converted = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    elif cs == "bgr":
        converted = resized_bgr.copy()
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    return resized_bgr, converted

def load_detect_and_preprocess_image(
        image_path: Path | str,
        target_size: Tuple[int, int] = (480, 360),
        color_space : str = "rgb",
        detect_circles: bool =False,
        min_radius_frac: float = 0.01,
        max_radius_frac: float = 0.12,
        padding_frac: float = 0.02,
        debug: bool = False
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    """
    Load, detect circles, resize and convert color space.

    Load -> Detect -> Crop -> Preprocess (Resize and color conversion).
    Returns: 
        - If detect_circles=false: (resized_bgr, converted_image)
        - If detect_circles=True: (cropped_resized_rgb, bbox_raw, centres) 
    """

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at path: {path.resolve()}")
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"OpenCV failed to decode image: {path.resolve()}")

    #Circle detection
    bbox_raw, centres, cropped_resized_rgb = None, None, None
    if detect_circles:
        img_rgb_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bbox_raw, centres = detect_four_circle_points(
            img_rgb_raw, min_radius_frac, max_radius_frac, padding_frac, debug
        )
        if bbox_raw is None or centres is None or len(centres) != 4:
            raise ValueError(
                f"Expected 4 Circles, got"
                f"{0 if centres is None else len(centres)}"
            )
        if debug:
            print(f"Raw bbox: {bbox_raw}, {len(centres)} centres detected")

        #crop
        x, y, w, h = bbox_raw
        cropped_rgb = img_rgb_raw[y:y+h, x:x+w]
        
        if debug:
            print(f"Cropped shape: {cropped_rgb.shape}")
        cropped_resized_rgb = cv2.resize(cropped_rgb, target_size, interpolation=cv2.INTER_AREA)
        if debug:
            print(f"GMM input image shape: {cropped_resized_rgb.shape}")

        return cropped_resized_rgb, bbox_raw, centres

    #Resize fallback
    resized_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    cs = color_space.lower()
    if cs == "rgb":
        converted = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    elif cs == "hsv":
        converted = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    elif cs == "bgr":
        converted = resized_bgr.copy()
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

    return resized_bgr, converted

