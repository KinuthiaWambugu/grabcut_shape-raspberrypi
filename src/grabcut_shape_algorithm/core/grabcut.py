import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class GrabCutSegmenter:
    """Applies morphological filtering on likelihood-derived seeds before running GrabCut."""

    def __init__(
        self,
        iterations: int = 5,
        border_margin: int = 8,
        morph_kernel_size: int = 5,
    ):
        self.iterations = iterations
        self.border_margin = border_margin
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

    def create_seed_mask(
            self,
            scene_img:np.ndarray,
            gmm,
            max_color_distance: float=100,
    )->np.ndarray:
        """Create GrabCut trimap and do morphological operations
        
        """
        #Scene GMM component assignments
        h, w = scene_img.shape[:2]
        pixels_scene = scene_img.reshape(-1,3).astype(np.float64)
        responsibilities = gmm.gmm_scene.predict_proba(pixels_scene)
        component_labels = np.argmax(
            responsibilities,
            axis=1
        ).reshape(h,w)
        component_confidence = np.max(
            responsibilities,
            axis=1
        ).reshape(h,w)

        #compute bg and scene component distances
        means_bg = gmm.gmm_bg.means_
        means_scene = gmm.gmm_scene.means_
        cost_matrix = cdist(
            means_bg,
            means_scene,
            metric = "euclidean"
        )

        #Hungarian matching with max color distance
        hungarian_cost = cost_matrix.copy()
        hungarian_cost[
            hungarian_cost > max_color_distance
        ] = 1e6
        row_ind, col_ind = linear_sum_assignment(hungarian_cost)
        valid_matches = []
        for bg_idx, scene_idx in zip(row_ind, col_ind):
            distance = cost_matrix[
                bg_idx,
                scene_idx
            ]
            if distance <= max_color_distance:
                valid_matches.append((bg_idx, scene_idx, distance))
        background_scene_components = { scene_idx
                                       for _, scene_idx, _ in valid_matches}
        all_scene_components = set(
            np.unique(component_labels)
        )
        foreground_scene_components = (all_scene_components - background_scene_components)

        raw_bg = np.isin(
            component_labels,
            list(background_scene_components)
        ).astype(np.uint8)*255

        raw_fg = np.isin(
            component_labels,
            list(foreground_scene_components)
        ).astype(np.uint8)*255

        return(
            raw_bg,
            raw_fg,
            component_labels,
            component_confidence,
            background_scene_components,
            foreground_scene_components,
            valid_matches,
            cost_matrix
        )

    def morphological_cleanup(
            self,
            raw_bg:np.ndarray,
            raw_fg:np.ndarray,
    )->np.ndarray:
        """
        Create raw GMM components masks and construct GrabCut trimap.
        """
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

        #clean fg
        clean_fg = cv2.morphologyEx(
            raw_fg, 
            cv2.MORPH_OPEN,
            small_kernel,
            iterations=1
        )
        clean_fg = cv2.morphologyEx(
            clean_fg,
            cv2.MORPH_CLOSE,
            cleanup_kernel,
            iterations = 1
        )

        #clean bg
        clean_bg = cv2.morphologyEx(
            raw_bg,
            cv2.MORPH_OPEN,
            small_kernel,
            iterations=1
        )

        #conservative definite fg/bg cores
        definite_fg = cv2.erode(clean_fg, small_kernel, iterations=1)
        definite_bg = cv2.erode(clean_bg, small_kernel, iterations=1)
        definite_fg[definite_bg == 255] = 0 #prevents overlap

        #assemble trimap
        h,w = raw_fg.shape
        mask = np.full(
            (h,w),
            cv2.GC_PR_BGD,
            dtype=np.uint8
        )
        #definite bg
        mask[definite_bg == 255] = cv2.GC_BGD
        mask[clean_fg == 255] = cv2.GC_PR_FGD
        mask[definite_fg == 255] = cv2.GC_FGD

        bm = self.border_margin
        mask[:bm, :] = cv2.GC_BGD
        mask[-bm:, :] = cv2.GC_BGD
        mask[:, :bm] = cv2.GC_BGD
        mask[:, -bm:] = cv2.GC_BGD

        return mask

    def run_grabcut(self, resized_bgr: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
        """Executes cv2.grabCut initialized with the morphologically cleaned mask."""
        mask = seed_mask.copy()
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        cv2.grabCut(
            resized_bgr,
            mask,
            None,
            bgd_model,
            fgd_model,
            iterCount=self.iterations,
            mode=cv2.GC_INIT_WITH_MASK,
        )

        return mask