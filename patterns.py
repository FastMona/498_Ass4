import json
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Any, Optional
import tkinter as tk

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install it with: pip install pillow"
    ) from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    plt = None

try:
    import numpy as np
except ImportError:
    np = None


THRESHOLD_LUT = [0] * 128 + [255] * 128


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = "patterns_orig"
PATTERN_DB = BASE_DIR / DEFAULT_OUTPUT_DIR / "patterns"
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def load_patterns() -> Dict[str, Dict[str, Any]]:
    if not PATTERN_DB.exists():
        return {}
    try:
        with PATTERN_DB.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if isinstance(data, dict):
        return data
    return {}


def save_patterns(data: Dict[str, Dict[str, Any]]) -> None:
    PATTERN_DB.parent.mkdir(parents=True, exist_ok=True)
    with PATTERN_DB.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_symbol(raw: str) -> str:
    value = raw.strip().upper()
    if len(value) != 1 or value not in ALLOWED_CHARS:
        raise ValueError("Use exactly one letter A-Z or one digit 0-9.")
    return value


def build_8x8_pattern(symbol: str) -> Image.Image:
    canvas = Image.new("L", (64, 64), 255)
    draw = ImageDraw.Draw(canvas)

    # Render large first, then downsample to 8x8 for a crisp bitmap pattern.
    try:
        font = ImageFont.truetype("arial.ttf", 54)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (64 - text_w) // 2
    y = (64 - text_h) // 2
    draw.text((x, y), symbol, fill=0, font=font)

    small = canvas.resize((8, 8), resample=Image.Resampling.LANCZOS)
    bw = small.point(THRESHOLD_LUT).convert("L")

    pixels = bw.load()
    if pixels is not None:
        # Ensure every row/column has at least one black pixel.
        for y in range(8):
            if all(pixels[x, y] == 255 for x in range(8)):
                pixels[3, y] = 0
        for x in range(8):
            if all(pixels[x, y] == 255 for y in range(8)):
                pixels[x, 3] = 0
    return bw


def save_pattern_image(symbol: str, pattern_img: Image.Image, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"pattern_{symbol}.jpg"
    pattern_img.save(image_path, format="JPEG", quality=100, subsampling=0)
    return image_path


def _set_window_title(fig, window_title: Optional[str]) -> None:
    manager = getattr(fig.canvas, "manager", None)
    if manager is None or not window_title:
        return
    try:
        manager.set_window_title(window_title)
    except Exception:
        return


def _show_pattern_window_process(image_array, symbol: str, window_title: Optional[str]) -> None:
    if plt is None:
        return
    cmap = ListedColormap(["white", "black"])
    fig, ax = plt.subplots(figsize=(6, 6))
    _set_window_title(fig, window_title)
    ax.imshow(image_array, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"8x8 Pattern for '{symbol}'")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.grid(color="black", linestyle="-", linewidth=0.5, alpha=0.25)
    fig.tight_layout()
    plt.show()


def _show_pattern_window_non_blocking(image: Image.Image, symbol: str) -> None:
    if np is None:
        return
    image_array = np.asarray(image, dtype=np.uint8)
    image_array = (image_array > 0).astype(np.uint8)

    context = get_context("spawn")
    process = context.Process(
        target=_show_pattern_window_process,
        args=(np.asarray(image_array), symbol, f"Pattern {symbol}"),
        daemon=True,
    )
    process.start()


def _show_gallery_window_process(image_arrays, image_titles, suptitle: str, window_title: Optional[str]) -> None:
    if plt is None:
        return
    cmap = ListedColormap(["white", "black"])
    fig, axes = plt.subplots(4, 5, figsize=(14, 10))
    _set_window_title(fig, window_title)

    flat_axes = axes.flatten()
    for i, axis in enumerate(flat_axes):
        if i < len(image_arrays):
            axis.imshow(image_arrays[i], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(image_titles[i])
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            axis.axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()


def _show_gallery_window_non_blocking(images, image_titles, suptitle: str, window_title: Optional[str]) -> None:
    if np is None:
        return

    image_arrays = [np.asarray(img, dtype=np.uint8) for img in images]
    image_arrays = [(arr < 128).astype(np.uint8) for arr in image_arrays]

    context = get_context("spawn")
    process = context.Process(
        target=_show_gallery_window_process,
        args=([np.asarray(arr) for arr in image_arrays], image_titles, suptitle, window_title),
        daemon=True,
    )
    process.start()


def show_pattern_in_figure(image_path: Path, symbol: str) -> None:
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return

    image = Image.open(image_path).convert("L")
    # Re-threshold before display to keep strict black/white rendering.
    image = image.point(THRESHOLD_LUT).convert("L")

    if plt is not None and np is not None:
        _show_pattern_window_non_blocking(image, symbol)
        return

    enlarged = image.resize((320, 320), Image.Resampling.NEAREST)
    root = tk.Tk()
    root.title(f"Pattern {symbol} (Tkinter)")
    photo = tk.PhotoImage(master=root, width=320, height=320)
    # Build a strict black/white preview by painting each scaled pixel block.
    pixels = enlarged.load()
    if pixels is None:
        print("Unable to read pixel data for display.")
        root.destroy()
        return
    rows = []
    for y in range(320):
        row = []
        for x in range(320):
            row.append("#000000" if pixels[x, y] == 0 else "#FFFFFF")
        rows.append("{" + " ".join(row) + "}")
    photo.put(" ".join(rows))

    image_refs = [photo]
    label = tk.Label(root, image=image_refs[0])
    label.pack(padx=10, pady=10)
    root.mainloop()


def create_pattern(patterns: Dict[str, Dict[str, Any]]) -> None:
    raw_symbol = input("Enter a single letter (A-Z) or digit (0-9): ")
    symbol = normalize_symbol(raw_symbol)

    folder_input = input("Enter folder to save pattern images (default patterns_orig): ").strip()
    folder_name = folder_input or DEFAULT_OUTPUT_DIR
    output_dir = BASE_DIR / folder_name

    pattern_img = build_8x8_pattern(symbol)
    image_path = save_pattern_image(symbol, pattern_img, output_dir)

    patterns[symbol] = {
        "symbol": symbol,
        "kind": "alphabet" if symbol.isalpha() else "number",
        "rows": 8,
        "cols": 8,
        "folder": str(output_dir.relative_to(BASE_DIR)),
        "file": image_path.name,
    }
    save_patterns(patterns)

    print(f"Pattern created: {image_path.name} (8x8 black/white)")


def edit_pattern(patterns: Dict[str, Dict[str, Any]]) -> None:
    raw_symbol = input("Enter symbol of pattern to edit (A-Z or 0-9): ")
    try:
        symbol = normalize_symbol(raw_symbol)
    except ValueError as e:
        print(e)
        return

    if symbol not in patterns:
        print(f"No saved pattern found for '{symbol}'.")
        return

    current = patterns[symbol]
    print(
        f"Current size: 8x8 | type: {current['kind']} | file: {current['file']}"
    )

    folder_input = input("Enter folder to save pattern images (default patterns_orig): ").strip()
    folder_name = folder_input or DEFAULT_OUTPUT_DIR
    output_dir = BASE_DIR / folder_name

    pattern_img = build_8x8_pattern(symbol)
    image_path = save_pattern_image(symbol, pattern_img, output_dir)

    patterns[symbol].update({
        "kind": "alphabet" if symbol.isalpha() else "number",
        "rows": 8,
        "cols": 8,
        "folder": str(output_dir.relative_to(BASE_DIR)),
        "file": image_path.name,
    })
    save_patterns(patterns)

    print(f"Pattern updated: {image_path.name} (8x8 black/white)")


def view_pattern(patterns: Dict[str, Dict[str, Any]]) -> None:
    orig_dir = BASE_DIR / DEFAULT_OUTPUT_DIR
    image_files = sorted(
        [p for p in orig_dir.glob("pattern_*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda p: p.stem,
    )

    if not image_files:
        print(f"No pattern images found in '{DEFAULT_OUTPUT_DIR}'.")
        return

    images = []
    image_titles = []
    for image_path in image_files[:20]:
        # Derive symbol from filename, e.g. pattern_A.png -> A
        stem = image_path.stem  # "pattern_A"
        symbol = stem.split("_", 1)[-1].upper() if "_" in stem else stem.upper()
        print(f"- {image_path.name}")
        image = Image.open(image_path).convert("L")
        image = image.point(THRESHOLD_LUT).convert("L")
        images.append(image)
        image_titles.append(symbol)

    if plt is not None and np is not None:
        print("Opening 4x5 gallery window for saved patterns...")
        _show_gallery_window_non_blocking(
            images,
            image_titles,
            "Saved 8x8 Patterns",
            "Pattern Gallery",
        )
        return

    print("Matplotlib/NumPy is unavailable; opening first pattern with Tkinter fallback.")
    show_pattern_in_figure(image_files[0], image_titles[0])


def main() -> None:
    patterns = load_patterns()

    while True:
        print("\nPattern Manager")
        print("1. Create pattern")
        print("2. Edit pattern")
        print("3. View pattern")
        print("4. Exit")

        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            try:
                create_pattern(patterns)
            except ValueError as e:
                print(f"Error: {e}")
        elif choice == "2":
            edit_pattern(patterns)
        elif choice == "3":
            view_pattern(patterns)
        elif choice == "4":
            print("Goodbye.")
            break
        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
