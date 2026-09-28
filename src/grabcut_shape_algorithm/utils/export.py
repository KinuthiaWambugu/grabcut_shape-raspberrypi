import cv2
import numpy as np


def extract_cutout(bgr_img: np.ndarray, grabcut_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generates a clean binary mask (0 or 255) and an RGBA image with alpha transparency."""
    # Values 1 (GC_FGD) and 3 (GC_PR_FGD) map to foreground
    binary_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Morphological closing to remove small isolated holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Construct RGBA transparent image
    b, g, r = cv2.split(bgr_img)
    rgba = cv2.merge([b, g, r, binary_mask])

    return binary_mask, rgba