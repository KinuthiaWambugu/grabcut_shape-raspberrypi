"""
Test gmm_priors using the following metrics because the mask could be an issue:
    1. Compare fitted gmms on a graph
    2. Bg-log likelihood graph.
    3. Gmm component matching using Hungarian algorithm and show if the gmm 
        components can be matched from bg and scene images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis, cdist
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "grabcut_shape_algorithm"))

from src.grabcut_shape_algorithm.core.gmm_priors import DualGMM
from src.grabcut_shape_algorithm.core.image_io import load_detect_and_preprocess_image, load_and_preprocess_image

class GMMTest:
    def __init__(self, seed: int = 42, target_size: tuple = (480, 360), output_dir:str = None):
        self.seed = seed
        self.target_size = target_size
        np.random.seed(seed)

        if output_dir is None:
            output_dir = f'data/gmm_test_results'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir created: {self.output_dir.resolve()}\n")

    def get_save_path(self, filename:str) ->str:
        return str(self.output_dir / filename)

    def test_fit_gmms(self, bg_img:np.ndarray, scene_img:np.ndarray)->DualGMM:
        gmm = DualGMM(k_bg=5, k_scene=12, max_samples=25000, random_state=self.seed)
        gmm.fit_bg(bg_img)
        gmm.fit_scene(scene_img)
        print("GMMs fitted well")
        print(f"BG GMM: {gmm.k_bg} components, scene GMM: {gmm.k_scene} components")
        return gmm

    def visualize_gmm_components(self, gmm:DualGMM, filename: str ="gmm_comparison.png"):
        save_path = self.get_save_path(filename)
        fig, axes = plt.subplots(1, 2, figsize = (14,6))

        ax = axes[0]
        means_bg = gmm.gmm_bg.means_
        covariances_bg = gmm.gmm_bg.covariances_
        ax.scatter(means_bg[:, 0], means_bg[:,1], s=100, c='blue',
                    marker='o', edgecolors='black', linewidth=2, label='BG Components', zorder=3)

        for i, (mean, cov) in enumerate(zip(means_bg, covariances_bg)):
            self._plot_ellipse(ax, mean[:2], cov[:2, :2], alpha=0.3, color='blue')

        ax.set_xlabel('Red channel', fontsize=11)
        ax.set_ylabel('Green channel', fontsize=11)
        ax.set_title(f"Background GMM ({gmm.k_bg} components)", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        #Scene GMM
        ax = axes[1]
        means_scene = gmm.gmm_scene.means_
        covariances_scene = gmm.gmm_scene.covariances_
        ax.scatter(means_scene[:, 0], means_scene[:, 1], s=100, c='red', marker='s',
                   edgecolors='black', linewidth=2, label='Scene components', zorder=3)
        for i, (mean,cov) in enumerate(zip(means_scene, covariances_scene)):
            self._plot_ellipse(ax, mean[:2], cov[:2,:2], alpha=0.3, color='red')
        ax.set_xlabel('Red channel', fontsize=11)
        ax.set_ylabel('Green channel', fontsize=11)
        ax.set_title(f'Scene GMM ({gmm.k_scene} components)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"GMM comparison plot saved to {save_path}")
        plt.show()

    def _plot_ellipse(self, ax, mean, cov, alpha=0.3, color='blue', n_std=2):
        "Plot covariance ellipse on axes"
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1,1], eigenvectors[0,1]))
        width, height = 4 * np.sqrt(eigenvalues) * n_std
        ellipse = Ellipse(mean, width, height, angle=angle, alpha=alpha,
                            facecolor=color, edgecolor=color, linewidth=1.5)
        ax.add_patch(ellipse)

    def compute_log_likelihood_maps(self, gmm:DualGMM, scene_img:np.ndarray)->dict:
        "Compute log-likelihood maps and return them"
        likelihoods = gmm.compute_loglikelihood_comparison(scene_img)
        return likelihoods

    def visualize_log_likelihood(self, likelihoods:dict, filename: str="log_likelihood_maps.png"):
        "Visualize bg log-likelihood and difference maps"
        save_path = self.get_save_path(filename)
        fig, axes = plt.subplots(1,3, figsize=(15,4))
        im0 = axes[0].imshow(likelihoods['bg_loglik'],cmap='viridis')
        axes[0].set_title('BG log-likelihood', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], label='Log likelihood')

        im1 = axes[1].imshow(likelihoods['scene_loglik'], cmap='viridis')
        axes[1].set_title("Scene log-likelihood", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], label='Log-likelihood')

        im2 = axes[2].imshow(likelihoods['diff_loglik'], cmap='RdBu_r')
        axes[2].set_title('Contrast (Scene - BG)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], label='Difference')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Log-likelihood maps saved to {save_path}")
        plt.show()

    def match_gmm_components(self, gmm:DualGMM) ->tuple[np.ndarray, np.ndarray, float]:
        """
        Match GMM components between bg and scene using Hungarian algo.
        Returns:  
            - matched_pairs: Array of (bg_idx, scene_idx) matched component pairs
            - cost_matrix: Cost matrix used for matching
            - total_cost: Total cost of optimal matching
        
        """
        means_bg = gmm.gmm_bg.means_
        means_scene = gmm.gmm_scene.means_
        
        # Compute pairwise distance matrix (cost matrix)
        cost_matrix = cdist(means_bg, means_scene, metric='euclidean')
        MAX_COLOR_DISTANCE = 100
        hungarian_cost = cost_matrix.copy()
        hungarian_cost[hungarian_cost > MAX_COLOR_DISTANCE] = 1e6
        row_ind, col_ind = linear_sum_assignment(hungarian_cost)
        valid_matches = []
        for bg_idx, scene_idx in zip(row_ind, col_ind):
            distance = cost_matrix[bg_idx, scene_idx]
            if distance <= MAX_COLOR_DISTANCE:
                valid_matches.append((bg_idx, scene_idx, distance))

        matched_pairs = np.array(
            [(bg_idx, scene_idx) for bg_idx, scene_idx, _ in valid_matches],
            dtype=int
        )
        total_cost = sum(
            distance for _,_, distance in valid_matches
        )

        print(f"\n Matches with color distance <= {MAX_COLOR_DISTANCE}:")
        for rank, (bg_idx, scene_idx, distance) in enumerate(valid_matches, 1):
            print(
                f"{rank:<6}"
                f"{bg_idx:<10}"
                f"{scene_idx:<11}"
                f"{distance:.4f}"
            )
        return matched_pairs, cost_matrix, total_cost

    def visualize_component_matching(self, gmm:DualGMM, matched_pairs:np.ndarray,
                                     cost_matrix: np.ndarray, filename:str = "component_matching.png"):

        save_path = self.get_save_path(filename)
        fig = plt.figure(figsize=(16, 5))
        
        # 1. Cost matrix heatmap
        ax1 = plt.subplot(1, 3, 1)
        im = ax1.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_xlabel('Scene GMM Component Index', fontsize=11)
        ax1.set_ylabel('Background GMM Component Index', fontsize=11)
        ax1.set_title('Component Matching Cost Matrix', fontsize=12, fontweight='bold')
        
        # Mark matched pairs
        for bg_idx, scene_idx in matched_pairs:
            ax1.plot(scene_idx, bg_idx, 'g*', markersize=15, markeredgecolor='white', markeredgewidth=1)
        
        plt.colorbar(im, ax=ax1, label='Euclidean Distance')
        
        # 2. Matching quality distribution
        ax2 = plt.subplot(1, 3, 2)
        matched_costs = cost_matrix[matched_pairs[:, 0], matched_pairs[:, 1]]
        ax2.bar(range(len(matched_costs)), matched_costs, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Matched Pair Index', fontsize=11)
        ax2.set_ylabel('Matching Cost', fontsize=11)
        ax2.set_title('Per-Pair Matching Costs', fontsize=12, fontweight='bold')
        ax2.axhline(matched_costs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {matched_costs.mean():.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Component matching on 2D projection
        ax3 = plt.subplot(1, 3, 3)
        means_bg = gmm.gmm_bg.means_
        means_scene = gmm.gmm_scene.means_
        
        # Plot BG components
        ax3.scatter(means_bg[:, 0], means_bg[:, 1], s=150, c='blue', marker='o', 
                   edgecolors='black', linewidth=2, label='BG Components', zorder=3)
        
        # Plot scene components
        ax3.scatter(means_scene[:, 0], means_scene[:, 1], s=150, c='red', marker='s', 
                   edgecolors='black', linewidth=2, label='Scene Components', zorder=3)
        
        # Draw matching lines
        for bg_idx, scene_idx in matched_pairs:
            ax3.plot([means_bg[bg_idx, 0], means_scene[scene_idx, 0]], 
                    [means_bg[bg_idx, 1], means_scene[scene_idx, 1]], 
                    'g--', alpha=0.5, linewidth=1.5, zorder=1)
        
        ax3.set_xlabel('Red Channel', fontsize=11)
        ax3.set_ylabel('Green Channel', fontsize=11)
        ax3.set_title('Component Matching Visualization', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Component matching visualization saved to {save_path}")
        plt.show()

    def print_gmm_weights_summary(self, gmm:DualGMM):
        print("\n" + "="*70)
        print("GMM component weights summary")
        print("="*70)

        #BG Weights
        print("\n Background GMM weights")
        print("-"*70)
        bg_weights = gmm.gmm_bg.weights_
        print(f"{'idx':<6} {'Weight': <12} {'Percentage':<12} {'Bar Chart':<40}")
        print("-"*70)

        for i, weight in enumerate(bg_weights):
            percentage = weight *100
            bar_length = int(weight*40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{i:<6} {weight:<12.4f} {percentage:<12.2f}% {bar}")

        print(f"\nTotal: {bg_weights.sum():.6f}")
        print(f"Max Weight: {bg_weights.max():.4f} (Component {np.argmax(bg_weights)})")
        print(f"Min Weight: {bg_weights.min():.4f} (Component {np.argmin(bg_weights)})")

        #Scene Weights
        print("\n" + "-" * 70)
        print("SCENE GMM WEIGHTS:")
        print("-" * 70)
        scene_weights = gmm.gmm_scene.weights_
        print(f"{'Idx':<6} {'Weight':<12} {'Percentage':<12} {'Bar Chart':<40}")
        print("-" * 70)
        
        for i, weight in enumerate(scene_weights):
            percentage = weight * 100
            bar_length = int(weight * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{i:<6} {weight:<12.4f} {percentage:<12.2f}% {bar}")
        
        print(f"\nTotal: {scene_weights.sum():.6f}")
        print(f"Max Weight: {scene_weights.max():.4f} (Component {np.argmax(scene_weights)})")
        print(f"Min Weight: {scene_weights.min():.4f} (Component {np.argmin(scene_weights)})")
        print("="*70 + "\n")


    def print_gmm_component_details(self, gmm: DualGMM):
        """Print detailed information about GMM components: weights and color (RGB) information."""
        print("\n" + "="*70)
        print("GMM COMPONENT DETAILS - WEIGHTS AND COLOR INFORMATION")
        print("="*70)
        
        # Background GMM
        print("\n" + "-"*70)
        print("BACKGROUND GMM COMPONENTS")
        print("-"*70)
        print(f"{'Idx':<4} {'Weight':<12} {'R (Mean)':<12} {'G (Mean)':<12} {'B (Mean)':<12}")
        print("-"*70)
        
        for i, (weight, mean) in enumerate(zip(gmm.gmm_bg.weights_, gmm.gmm_bg.means_)):
            r_mean, g_mean, b_mean = mean[0], mean[1], mean[2]
            print(f"{i:<4} {weight:<12.4f} {r_mean:<12.2f} {g_mean:<12.2f} {b_mean:<12.2f}")
        
        print(f"\nTotal Weight (should be 1.0): {gmm.gmm_bg.weights_.sum():.6f}")
        
        # Scene GMM
        print("\n" + "-"*70)
        print("SCENE GMM COMPONENTS")
        print("-"*70)
        print(f"{'Idx':<4} {'Weight':<12} {'R (Mean)':<12} {'G (Mean)':<12} {'B (Mean)':<12}")
        print("-"*70)
        
        for i, (weight, mean) in enumerate(zip(gmm.gmm_scene.weights_, gmm.gmm_scene.means_)):
            r_mean, g_mean, b_mean = mean[0], mean[1], mean[2]
            print(f"{i:<4} {weight:<12.4f} {r_mean:<12.2f} {g_mean:<12.2f} {b_mean:<12.2f}")
        
        print(f"\nTotal Weight (should be 1.0): {gmm.gmm_scene.weights_.sum():.6f}")
        print("="*70 + "\n")

    def print_comp_matching_summary(self, gmm: DualGMM, matched_pairs: np.ndarray, cost_matrix: np.ndarray):
        """Print detailed summary of component matching with color information."""
        print("\n" + "="*70)
        print("COMPONENT MATCHING SUMMARY (Hungarian Algorithm)")
        print("="*70)
        
        matched_costs = cost_matrix[matched_pairs[:, 0], matched_pairs[:, 1]]
        
        print(f"\nMatching Statistics:")
        print(f"  Total cost:        {matched_costs.sum():.4f}")
        print(f"  Mean cost:         {matched_costs.mean():.4f}")
        print(f"  Std dev:           {matched_costs.std():.4f}")
        print(f"  Min cost:          {matched_costs.min():.4f}")
        print(f"  Max cost:          {matched_costs.max():.4f}")
        
        print(f"\nTop 5 Best Matches (lowest cost):")
        print(f"{'Rank':<6} {'BG Idx':<8} {'Scene Idx':<10} {'Cost':<8} {'BG Color (R,G,B)':<20} {'Scene Color (R,G,B)':<20}")
        print("-" * 72)
        best_idx = np.argsort(matched_costs)[:5]
        for rank, idx in enumerate(best_idx, 1):
            bg_idx, scene_idx = matched_pairs[idx]
            cost = matched_costs[idx]
            bg_color = gmm.gmm_bg.means_[bg_idx]
            scene_color = gmm.gmm_scene.means_[scene_idx]
            bg_color_str = f"({bg_color[0]:.1f},{bg_color[1]:.1f},{bg_color[2]:.1f})"
            scene_color_str = f"({scene_color[0]:.1f},{scene_color[1]:.1f},{scene_color[2]:.1f})"
            print(f"{rank:<6} {bg_idx:<8} {scene_idx:<10} {cost:<8.4f} {bg_color_str:<20} {scene_color_str:<20}")
        
        print(f"\nTop 5 Worst Matches (highest cost):")
        print(f"{'Rank':<6} {'BG Idx':<8} {'Scene Idx':<10} {'Cost':<8} {'BG Color (R,G,B)':<20} {'Scene Color (R,G,B)':<20}")
        print("-" * 72)
        worst_idx = np.argsort(matched_costs)[-5:][::-1]
        for rank, idx in enumerate(worst_idx, 1):
            bg_idx, scene_idx = matched_pairs[idx]
            cost = matched_costs[idx]
            bg_color = gmm.gmm_bg.means_[bg_idx]
            scene_color = gmm.gmm_scene.means_[scene_idx]
            bg_color_str = f"({bg_color[0]:.1f},{bg_color[1]:.1f},{bg_color[2]:.1f})"
            scene_color_str = f"({scene_color[0]:.1f},{scene_color[1]:.1f},{scene_color[2]:.1f})"
            print(f"{rank:<6} {bg_idx:<8} {scene_idx:<10} {cost:<8.4f} {bg_color_str:<20} {scene_color_str:<20}")
        
        print("\n" + "="*70 + "\n")

def run_full_test_suite(bg_image_path: str, scene_image_path: str, target_size: tuple = (480, 360)):
    """Run complete test suite with all three metrics.
    
    Args:
        bg_image_path: Path to background image
        scene_image_path: Path to scene image
        target_size: Target image size as (width, height)
    """
    print("\n" + "="*70)
    print("DUAL GMM TEST SUITE - COMPREHENSIVE EVALUATION")
    print("="*70 + "\n")
    
    # Initialize test suite
    tester = GMMTest(seed=42, target_size=target_size)
    
    # Load and preprocess images
    print(f"Loading and preprocessing images to size {target_size}...")
    #bg image
    bg_img, bg_bbox, bg_centres = load_detect_and_preprocess_image(
        bg_image_path,
        target_size=target_size,
        color_space="rgb",
        detect_circles=True,
        debug=True
    )
    #_, bg_img = load_and_preprocess_image(bg_image_path, target_size=target_size, color_space="rgb")
    scene_img, bbox_raw, centres = load_detect_and_preprocess_image(scene_image_path, target_size=target_size, color_space="rgb", detect_circles=True, debug=True)
    print(f"[✓] Background image loaded: {bg_img.shape} (RGB)")
    print(f"[✓] Scene image loaded: {scene_img.shape} (RGB)\n")
    
    # Test 1: Fit GMMs
    print("TEST 1: Fitting Dual GMMs")
    print("-" * 70)
    gmm = tester.test_fit_gmms(bg_img, scene_img)
    print()
    tester.print_gmm_weights_summary(gmm)
    tester.print_gmm_component_details(gmm)
    
    # Test 2: Visualize GMM Components
    print("TEST 2: GMM Component Comparison")
    print("-" * 70)
    tester.visualize_gmm_components(gmm, filename="gmm_comparison.png")
    print()
    
    # Test 3: Compute and Visualize Log-Likelihood
    print("TEST 3: Background Log-Likelihood Analysis")
    print("-" * 70)
    likelihoods = tester.compute_log_likelihood_maps(gmm, scene_img)
    tester.visualize_log_likelihood(likelihoods, filename="log_likelihood_maps.png")
    print()
    
    # Test 4: Component Matching with Hungarian Algorithm
    print("TEST 4: GMM Component Matching (Hungarian Algorithm)")
    print("-" * 70)
    matched_pairs, cost_matrix, total_cost = tester.match_gmm_components(gmm)
    tester.visualize_component_matching(gmm, matched_pairs, cost_matrix, 
                                        filename="component_matching.png")
    tester.print_comp_matching_summary(gmm, matched_pairs, cost_matrix)
    
    print("="*70)
    print("TEST SUITE COMPLETE")
    print("="*70 + "\n")
 
 
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python test_gmm_priors.py <bg_image_path> <scene_image_path> [width] [height]")
        print("\nExample:")
        print("  python test_gmm_priors.py background.jpg scene.jpg")
        print("  python test_gmm_priors.py background.jpg scene.jpg 480 360")
        sys.exit(1)
    
    bg_path = sys.argv[1]
    scene_path = sys.argv[2]
    
    # Optional target size
    target_size = (480, 360)  # default (width, height)
    if len(sys.argv) >= 5:
        target_size = (int(sys.argv[3]), int(sys.argv[4]))
    
    run_full_test_suite(bg_path, scene_path, target_size=target_size)

        
        




