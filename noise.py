from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiprocessing import get_context
import numpy as np


GRID_ROWS: int = 8
GRID_COLS: int = 8

NOISY_PATTERNS_DIR = Path(__file__).resolve().parent / "patterns_noisy"


def ensure_patterns_dir() -> Path:
    """Return the default patterns directory path."""
    return Path(__file__).resolve().parent / "patterns_orig"


def load_pattern_image(image_path: Path) -> list[list[int]] | None:
    """Load a binary pattern image and return a 2D list of 0/1 values."""
    raw = plt.imread(str(image_path))
    # Normalise to float [0, 1]; jpeg comes in as uint8 0-255
    if raw.dtype == np.uint8:
        raw = raw / 255.0
    # Collapse RGB/RGBA to single channel by averaging
    if raw.ndim == 3:
        raw = raw.mean(axis=2)
    if raw.shape != (GRID_ROWS, GRID_COLS):
        # Nearest-neighbour resize using numpy block averaging is not needed;
        # just verify size matches expectation
        return None
    # Dark pixels (low value) represent ink -> 1; white (high value) -> 0
    return (raw < 0.5).astype(int).tolist()


def _set_window_title(fig, window_title: str | None) -> None:
    """Set matplotlib window title when backend supports it."""
    manager = getattr(fig.canvas, "manager", None)
    if manager is None or not window_title:
        return
    try:
        manager.set_window_title(window_title)
    except Exception:
        return


def _show_gallery_window_process(
    images: list[np.ndarray],
    image_titles: list[str],
    suptitle: str,
    window_title: str | None,
) -> None:
    """Render up to 8 images in a 2x4 gallery in a child process."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    _set_window_title(fig, window_title)
    axes_flat = axes.flatten()
    cmap = ListedColormap(["white", "black"])

    for index, axis in enumerate(axes_flat):
        if index >= len(images):
            axis.axis("off")
            continue

        image = images[index]
        if image.ndim == 3:
            axis.imshow(image)
        else:
            axis.imshow(image, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(image_titles[index])
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def show_gallery_window(
    images: list[np.ndarray],
    image_titles: list[str],
    suptitle: str,
    window_title: str | None = None,
) -> None:
    """Start a process that shows a gallery window."""
    context = get_context("spawn")
    process = context.Process(
        target=_show_gallery_window_process,
        args=(
            [np.asarray(image) for image in images],
            image_titles,
            suptitle,
            window_title,
        ),
        daemon=True,
    )
    process.start()


def ensure_noisy_patterns_dir() -> Path:
    """Ensure noisy patterns output folder exists and return its path."""
    NOISY_PATTERNS_DIR.mkdir(exist_ok=True)
    return NOISY_PATTERNS_DIR


def confirm_yes_no_default_no(prompt: str) -> bool:
    """Return True only when user explicitly answers yes; default is no."""
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def delete_image_files_in_folder(folder: Path) -> int:
    """Delete all PNG/JPG image files in the given folder and return count deleted."""
    if not folder.exists() or not folder.is_dir():
        return 0

    deleted_count = 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for image_file in folder.glob(ext):
            image_file.unlink(missing_ok=True)
            deleted_count += 1
    return deleted_count


def read_noise_percent() -> float:
    """Read noise percent in range [0, 100]."""
    while True:
        value = input("Enter percent noise (0-100): ").strip()
        try:
            noise_percent = float(value)
        except ValueError:
            print("Invalid input: enter a number between 0 and 100.")
            continue

        if 0 <= noise_percent <= 100:
            return noise_percent
        print("Invalid input: percent noise must be between 0 and 100.")


def apply_noise_to_grid(grid: list[list[int]], noise_percent: float) -> list[list[int]]:
    """Randomly flip approximately noise_percent of bits in the grid."""
    grid_array = np.asarray(grid, dtype=np.uint8)
    total_bits = grid_array.size
    flip_count = int(round((noise_percent / 100.0) * total_bits))

    if flip_count <= 0:
        return grid_array.tolist()

    rng = np.random.default_rng()
    flat_indices = rng.choice(total_bits, size=flip_count, replace=False)
    flat_grid = grid_array.reshape(-1)
    flat_grid[flat_indices] = 1 - flat_grid[flat_indices]
    return flat_grid.reshape((GRID_ROWS, GRID_COLS)).tolist()


def display_recent_noisy_patterns(folder: Path, noise_percent_text: str) -> None:
    """Display a 2x4 figure of the 8 most recently generated noisy images."""
    recent_files = sorted(
        list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )[:8]

    if not recent_files:
        return

    images: list[np.ndarray] = []
    image_titles: list[str] = []

    for image_path in recent_files:
        images.append(np.asarray(plt.imread(str(image_path))))
        image_titles.append(image_path.stem)

    ordered_items = sorted(zip(image_titles, images), key=lambda item: item[0].lower())
    image_titles = [item[0] for item in ordered_items]
    images = [item[1] for item in ordered_items]

    show_gallery_window(
        images,
        image_titles,
        suptitle=f"{noise_percent_text}% Noisy Patterns ({GRID_ROWS} X {GRID_COLS})",
        window_title="Most Recent Noisy Patterns",
    )


def run_create_noisy_patterns() -> None:
    """Create noisy versions of all patterns from a selected source folder."""
    noise_percent = read_noise_percent()
    if noise_percent.is_integer():
        noise_percent_text = str(int(noise_percent))
        noise_label = str(int(noise_percent))
    else:
        noise_percent_text = str(noise_percent).rstrip("0").rstrip(".")
        noise_label = noise_percent_text.replace(".", "p")

    default_folder = ensure_patterns_dir()
    source_input = input(f"Folder to noisify [{default_folder.name}]: ").strip()
    source_folder = Path(source_input) if source_input else default_folder

    if not source_folder.is_absolute():
        source_folder = Path(__file__).resolve().parent / source_folder

    if not source_folder.exists() or not source_folder.is_dir():
        print(f"Folder not found: {source_folder}")
        return

    destination_folder = ensure_noisy_patterns_dir()

    deleted_count = delete_image_files_in_folder(destination_folder)
    if deleted_count:
        print(f"Deleted {deleted_count} existing image file(s) from {destination_folder.name}.")

    source_images = sorted(
        list(source_folder.glob("*.png"))
        + list(source_folder.glob("*.jpg"))
        + list(source_folder.glob("*.jpeg"))
    )
    if not source_images:
        print(f"No image patterns found in: {source_folder}")
        return

    created_count = 0
    for source_image in source_images:
        grid = load_pattern_image(source_image)
        if grid is None:
            print(f"Skipping {source_image.name}: expected {GRID_COLS}x{GRID_ROWS} binary pattern image.")
            continue

        noisy_grid = apply_noise_to_grid(grid, noise_percent)
        output_path = destination_folder / f"{noise_label}n_{source_image.name}"
        grid_array = np.asarray(noisy_grid, dtype=np.uint8)
        cmap = ListedColormap(["white", "black"])
        plt.imsave(str(output_path), grid_array, cmap=cmap, vmin=0, vmax=1)
        created_count += 1
        print(f"Created: {output_path.name}")

    print(f"Noisy pattern generation complete: {created_count} file(s) in {destination_folder}")

    if created_count <= 0:
        return

    display_recent_noisy_patterns(destination_folder, noise_percent_text)
