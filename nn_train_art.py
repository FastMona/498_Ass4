"""Training program for Fuzzy ARTMAP classifier.

Depends on first_ai.data/build_mnist_dataloaders. Autoencoder training and
OOD parameter computation are handled separately.
"""

import logging
import os
import time
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn, save
from torch.utils.data import DataLoader

from config import Config
from nn_model_art import FuzzyARTMAPClassifier

# Configure logging to display output
logging.basicConfig(level=logging.INFO, format='%(message)s')

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.append(str(SRC_DIR))

from first_ai.data import build_mnist_dataloaders  # type: ignore
from first_ai.logging_utils import get_environment_info, log_environment_block  # type: ignore

logger = logging.getLogger(__name__)


def display_art_category_distribution(art):
    """Display Fuzzy ART category distribution across digits."""
    
    logger.info("\n" + "=" * 80)
    logger.info("Fuzzy ART Category Distribution Analysis")
    logger.info("=" * 80)

    # Get category labels and counts
    category_labels = art.category_labels.cpu()
    category_counts = art.category_counts.cpu()
    committed = art.committed.cpu()

    # Count categories per digit
    digit_categories = {i: [] for i in range(10)}

    for cat_idx in range(art.max_categories):
        if committed[cat_idx]:
            label = category_labels[cat_idx].item()
            if 0 <= label < 10:
                count = category_counts[cat_idx].item()
                digit_categories[label].append({
                    'category_id': cat_idx,
                    'pattern_count': count
                })

    total_categories = sum(1 for c in committed if c.item())
    logger.info(
        f"Yes, the {art.max_categories} categories are mapped to the 10 digits (0-9). Here's the breakdown:\n"
    )

    logger.info(f"{'Digit':<6}{'# Categories':<14}{'Total Patterns':<16}Notable Patterns")
    logger.info("-" * 80)

    per_digit_stats = []
    for digit in range(10):
        cats = digit_categories[digit]
        num_cats = len(cats)
        counts = [c['pattern_count'] for c in cats]
        total_patterns = sum(counts)

        if counts:
            max_count = max(counts)
            min_count = min(counts)
            max_cat = cats[counts.index(max_count)]['category_id']
            sorted_counts = sorted(counts, reverse=True)
            second_max = sorted_counts[1] if len(sorted_counts) > 1 else 0
            balance_ratio = (max_count / max(min_count, 1)) if min_count > 0 else float('inf')

            if max_count >= 0.8 * total_patterns and num_cats > 1:
                notable = f"1 mega-category (C{max_cat}: {max_count:,}) + {num_cats - 1} smaller ones"
            elif max_count >= 3 * max(second_max, 1):
                notable = f"1 dominant category (C{max_cat}: {max_count:,}) + others"
            elif num_cats <= 6 and balance_ratio <= 1.5:
                notable = "Smallest overall, most uniform distribution"
            elif balance_ratio <= 1.7:
                notable = f"Well-balanced categories (~{min_count:,}-{max_count:,} patterns each)"
            else:
                notable = "Moderate distribution"
        else:
            notable = "No committed categories"

        logger.info(f"{digit:<6}{num_cats:<14}{total_patterns:<16}{notable}")
        per_digit_stats.append({
            'digit': digit,
            'num_cats': num_cats,
            'total_patterns': total_patterns,
            'counts': counts,
        })

    logger.info("\nKey Findings:")
    all_committed = " (all committed)" if total_categories == art.max_categories else ""
    avg_categories = sum(d['num_cats'] for d in per_digit_stats) / 10
    min_categories = min(d['num_cats'] for d in per_digit_stats)
    max_categories = max(d['num_cats'] for d in per_digit_stats)
    total_patterns_all = sum(d['total_patterns'] for d in per_digit_stats)

    most_simple = min(per_digit_stats, key=lambda d: d['num_cats'])['digit']
    most_complex = max(per_digit_stats, key=lambda d: d['num_cats'])['digit']

    dominant_digits = []
    for d in per_digit_stats:
        if d['counts']:
            top = max(d['counts'])
            if top >= 0.8 * d['total_patterns'] and d['num_cats'] > 1:
                dominant_digits.append(d['digit'])

    logger.info(f"Total categories used: {total_categories}/{art.max_categories}{all_committed}")
    logger.info(f"Average: {avg_categories:.1f} categories per digit")
    if dominant_digits:
        digits_list = ", ".join(str(d) for d in dominant_digits)
        logger.info(
            "Imbalance: digits with a dominant category detected: "
            f"{digits_list}"
        )
    else:
        logger.info("Imbalance: no dominant categories detected")
    logger.info(f"Most balanced: digits with {min_categories}-{max_categories} categories")
    logger.info(
        f"Digit {most_simple} is the simplest ({min_categories} categories) while "
        f"Digit {most_complex} is most complex ({max_categories} categories)"
    )
    logger.info(f"Total patterns learned: {total_patterns_all:,}")

    logger.info("=" * 80 + "\n")
def get_optimal_num_workers(requested_workers: int = None) -> int:
    """
    Determine optimal num_workers based on available CPU cores.
    
    Rule of thumb for PyTorch DataLoader:
    - num_workers should be <= cpu_count (avoid context switching overhead)
    - More workers = better prefetching and data loading parallelism
    - Typical range: 4-16 depending on system size
    - Sweet spot: ~cpu_count // 4 to cpu_count // 2
    - Cap at 12 for practical diminishing returns
    
    Args:
        requested_workers: User-requested num_workers (None = auto-detect)
    
    Returns:
        Optimal num_workers for this hardware
    """
    cpu_count = os.cpu_count() or 1
    
    if requested_workers is not None:
        # If user explicitly requests workers, cap it at CPU count
        optimal = min(requested_workers, cpu_count)
    else:
        # Auto-detect: use aggressive prefetching
        # Formula: cpu_count // 3, capped at 12 for practical reasons
        # Results: 4-core→1, 8-core→2, 12-core→4, 24-core→8, 48-core→12
        optimal = max(2, min(12, cpu_count // 3))
    
    return optimal


def get_cpu_info() -> dict:
    """Get CPU information for logging."""
    cpu_count = os.cpu_count() or 1
    return {
        'total_cores': cpu_count,
        'logical_processors': cpu_count,  # Same on most systems
    }


def train_art(
    art,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    num_passes: int = 3,
) -> None:
    art.train()
    total_batches = len(train_loader)

    for pass_num in range(num_passes):
        logger.info(f"\n{'='*80}")
        logger.info(f"  PASS {pass_num + 1}/{num_passes} - Processing {len(train_loader.dataset)} training samples")
        logger.info(f"{'='*80}")

        total_samples = 0
        pass_start_time = time.time()
        batch_times = []

        for batch_idx, (X, y) in enumerate(train_loader):
            batch_start = time.time()
            # Keep data on CPU for ART - GPU transfers are more expensive than CPU processing
            # ART is inherently sequential with limited parallelization opportunities
            X, y = X.to('cpu'), y.to('cpu')

            for i in range(X.size(0)):
                art.train_pattern(X[i].view(-1), y[i].item())
                total_samples += 1

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if (batch_idx + 1) % 50 == 0:
                avg_batch_time = np.mean(batch_times[-50:])
                samples_per_sec = (50 * train_loader.batch_size) / sum(batch_times[-50:])
                progress_pct = (batch_idx + 1) / total_batches * 100
                eta_batches = total_batches - (batch_idx + 1)
                eta_seconds = eta_batches * avg_batch_time
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                current_time = time.strftime("%H:%M")
                display_batch = (batch_idx + 1) // 50
                display_total_batches = math.ceil(total_batches / 50)

                logger.info(
                    f"{progress_pct:.1f}% complete | Batch {display_batch}/{display_total_batches} | "
                    f"Categories {art.num_committed}/{art.max_categories} | Speed: {samples_per_sec:.1f} samp/sec | "
                    f"Time remaining: {eta_min:02d}:{eta_sec:02d} | {current_time}"
                )

        pass_time = time.time() - pass_start_time
        pass_min = int(pass_time // 60)
        pass_sec = int(pass_time % 60)

        logger.info(f"\n✓ Pass {pass_num + 1} complete in {pass_min}m {pass_sec}s")
        logger.info(f"  Total samples processed: {total_samples}")
        logger.info(f"  Categories committed: {art.num_committed}/{Config.ART_MAX_CATEGORIES}")
        logger.info(f"  Average speed: {total_samples/pass_time:.1f} samples/s")

        # Evaluate after each pass
        art.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in test_loader:
                # Evaluate on CPU where ART runs
                X, y = X.to('cpu'), y.to('cpu')
                X_flat = X.view(X.size(0), -1)
                logits = art.predict(X_flat)
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = 100 * correct / total
        logger.info(f"  Test Accuracy after pass {pass_num + 1}: {accuracy:.2f}% ({correct}/{total} correct)")
        art.train()


def calibrate_reconstruction_threshold_art(autoencoder, art, val_loader, device: str = "cpu"):
    autoencoder.eval()
    art.eval()
    recon_errors = []

    with torch.no_grad():
        for X, _ in val_loader:
            # Move to CPU for processing with ART model
            X = X.to('cpu')
            X_flat = X.view(X.size(0), -1)
            preds = torch.argmax(art.predict(X_flat), dim=1)
            errors = autoencoder.reconstruction_error(X, preds)
            recon_errors.extend(errors.cpu().tolist())

    recon_errors = np.array(recon_errors)
    recon_threshold_95 = float(np.percentile(recon_errors, 95))
    recon_threshold_99 = float(np.percentile(recon_errors, 99))
    recon_mean = float(np.mean(recon_errors))
    recon_std = float(np.std(recon_errors))

    logger.info("\nReconstruction error statistics on VALIDATION data:")
    logger.info(f"  Samples: {len(recon_errors)}")
    logger.info(f"  Mean: {recon_mean:.6f}")
    logger.info(f"  Std: {recon_std:.6f}")
    logger.info(f"  95th percentile: {recon_threshold_95:.6f}")
    logger.info(f"  99th percentile: {recon_threshold_99:.6f}")

    return recon_threshold_95, recon_threshold_99, recon_mean, recon_std


def main(
    device: str = "auto",
    train_batch_size: int = 64,
    eval_batch_size: int = 256,
    num_workers: int = None,
    passes: int = 1,  # CRITICAL: Single-pass only! Multi-pass causes template collapse to universal attractors
    sort_data: bool = False,  # If True, train on sorted data (grouped by digit) to test category accumulation
):
    # ART uses CPU for training (GPU transfers are expensive for sequential processing)
    # Set device to CPU since GPU doesn't improve ART performance
    device = 'cpu'
    
    # Auto-detect optimal num_workers based on CPU count
    if num_workers is None:
        num_workers = get_optimal_num_workers()
    else:
        num_workers = get_optimal_num_workers(num_workers)
    
    cpu_info = get_cpu_info()

    env_info = get_environment_info()
    log_environment_block(logger, env_info)

    logger.info("\n" + "=" * 80)
    logger.info("  Training Fuzzy ARTMAP Network".center(80))
    logger.info("=" * 80)
    logger.info("\n🚀 Training setup:")
    logger.info(f"  • Training device: {device.upper()} (ART uses CPU for sequential updates)")
    logger.info(f"  • Train batch size: {train_batch_size}")
    logger.info(f"  • Eval batch size: {eval_batch_size}")
    logger.info(f"  • Data workers: {num_workers}")
    logger.info(f"  • Passes: {passes}")
    logger.info(f"  • Data order: {'sorted by digit' if sort_data else 'random'}")
    logger.info(f"  • CPU cores available: {cpu_info['total_cores']}")
    logger.info(f"\n🔧 ARTMAP Architecture:")
    logger.info(f"  • Max categories: {Config.ART_MAX_CATEGORIES}")
    logger.info(f"  • Vigilance (ρ): {Config.ART_VIGILANCE}")
    logger.info(f"  • Learning rate (β): {Config.ART_LEARNING_RATE}")
    logger.info(f"  • Choice parameter (α): {Config.ART_CHOICE_ALPHA}")
    logger.info(f"  • Count penalty (γ): {Config.ART_COUNT_PENALTY_GAMMA}")
    logger.info(f"  • Max per category: {Config.ART_MAX_CATEGORY_COUNT}")
    logger.info(f"  • Match tracking ε: {Config.ART_MATCH_TRACKING_EPS}")

    train_loader, val_loader, test_loader = build_mnist_dataloaders(
        dataset_root=Path("training_data"),
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        train_ratio=getattr(Config, "TRAIN_RATIO", 0.8),
        seed=getattr(Config, "RANDOM_SEED", 42),
        sort_by_label=sort_data,
    )

    art = FuzzyARTMAPClassifier(
        input_dim=Config.INPUT_SIZE * Config.INPUT_SIZE,
        max_categories=Config.ART_MAX_CATEGORIES,
        vigilance=Config.ART_VIGILANCE_SORTED if sort_data else Config.ART_VIGILANCE,
        learning_rate=Config.ART_LEARNING_RATE_SORTED if sort_data else Config.ART_LEARNING_RATE,
        choice_alpha=Config.ART_CHOICE_ALPHA,
        count_penalty_gamma=Config.ART_COUNT_PENALTY_GAMMA,
        max_category_count=Config.ART_MAX_CATEGORY_COUNT,
        match_tracking_epsilon=Config.ART_MATCH_TRACKING_EPS,
    ).to('cpu')  # Keep ART on CPU - GPU transfers are slower than CPU sequential processing

    # ART training
    train_art(art, train_loader, test_loader, device=device, num_passes=passes)

    logger.info(f"\n✓ Saving ART model to {Config.MODEL_PATH_ART}")
    with open(Config.MODEL_PATH_ART, "wb") as f:
        save(art.state_dict(), f)

    # Display category distribution analysis
    display_art_category_distribution(art)

    logger.info("\n" + "=" * 80)
    logger.info("  ART Training Complete!".center(80))
    logger.info("=" * 80)
    logger.info(f"  - {Config.MODEL_PATH_ART} (ART classifier)")
    logger.info("  - Next: Train CCA and compute OOD params (separate options)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Fuzzy ARTMAP for MNIST")
    parser.add_argument("--passes", type=int, default=1, help="Number of training passes (default: 1)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (default: auto)")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size (default: 64)")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Evaluation batch size (default: 256)")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of data workers (default: auto)")
    parser.add_argument("--sort-data", action="store_true", help="Train on sorted data (grouped by digit)")
    
    args = parser.parse_args()
    
    main(
        device=args.device,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        passes=args.passes,
        sort_data=args.sort_data,
    )
