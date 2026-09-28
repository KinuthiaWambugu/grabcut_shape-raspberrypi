import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from src.grabcut_shape_algorithm.core.image_io import load_detect_and_preprocess_image, detect_four_circle_points

image_path = "data/raw/Scene4.jpeg"
output_dir = Path("data/test_images")
output_dir.mkdir(parents=True, exist_ok=True)

start = time.perf_counter()

cropped_resized_bgr, bbox_raw, circle_centres= load_detect_and_preprocess_image(
        image_path,
        target_size=(480, 360),
        color_space="rgb",
        detect_circles=True,
        debug=True
    )

result = load_detect_and_preprocess_image(image_path, target_size=(480,360), color_space="rgb", detect_circles=True, debug=True)
print(f"Number of return values: {len(result)}")
print(f"Return value types:{[type(r) for r in result]}")

elapsed = time.perf_counter() - start
cx_x, cx_y, cx_w, cx_h = bbox_raw
print(f"Circle detected")
print(f"Circle-based bbox: x={cx_x}, y={cx_y}, w={cx_w}, h={cx_h}")
print(f" Detected centres: {circle_centres.tolist()}")
print(f"Time elapsed: {elapsed:.3f}s")

fig, axes = plt.subplots(1,2,figsize=(14,6))
img_demo = cv2.imread(image_path)
img_demo = cv2.cvtColor(img_demo, cv2.COLOR_BGR2RGB)

ax=axes[0]
ax.imshow(img_demo)
bx, by, bw, bh = bbox_raw
ax.add_patch(plt.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor="lime",
                           facecolor="none", label="detection bbox"))
ax.scatter(circle_centres[:,0], circle_centres[:,1],
           c="red", s=80, zorder=5, marker="x", linewidth=2, label="Detected centres")
ax.set_title(f"Raw Image detection")
ax.legend(fontsize=10)
ax.axis("off")

ax=axes[1]
ax.imshow(cropped_resized_bgr)
ax.set_title(f"GMM input (480* 360)")
ax.axis("off")
plt.suptitle(f"Circle detection and preprocessing ({elapsed:.3f}s")
plt.tight_layout()
viz_path = output_dir / "test_detection_vis.png"
plt.savefig(viz_path, dpi=150, bbox_inches="tight")
print("\nSaved test vis: {viz_path}")
plt.show()


