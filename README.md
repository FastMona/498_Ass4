# 498_Ass4

Neural-network assignment project for recognizing symbols **A-T** from 8x8 black/white patterns using three ART-family models:

- ART1
- Fuzzy ART
- Augmented Fuzzy ART

The repository includes:

- Pattern creation/editing tools
- Model training and evaluation scripts
- A dashboard that ties the workflow together

## Project Structure

- `dash.py`: main interactive dashboard for the assignment workflow
- `nn_model_art.py`: shared model definitions and model I/O
- `nn_train_art.py`: training/evaluation pipeline and CLI training entrypoint
- `patterns.py`: 8x8 pattern manager (create, edit, preview)
- `patterns_orig/`: default source pattern images
- `patterns_Ass4/`: saved trained model JSON files
- `path_memory.json`: remembers previously entered paths in the dashboard
- `terminal.txt`: captured dashboard terminal session output

## Requirements

- Python 3.10+
- Pip packages:
  - `pillow`
  - `numpy` (optional, recommended for plotting windows)
  - `matplotlib` (optional, recommended for plotting windows)

Install dependencies:

```bash
pip install pillow numpy matplotlib
```

## Quick Start

Run the dashboard:

```bash
python dash.py
```

From the dashboard menu, you can:

1. Manage pattern images
2. Create/train one or more ART models
3. Sweep vigilance values for a trained model
4. Recognize a single image
5. Recognize a folder of images
6. Show model summary table

## Command-Line Training

You can train directly without the dashboard:

```bash
python nn_train_art.py --model-type all --pattern-dir patterns_orig --model-dir patterns_Ass4
```

Useful options:

- `--model-type`: `all`, `art1`, `fuzzy_art`, `aug_fuzzy_art`
- `--epochs`: training epochs (default `20`)
- `--augment-per-symbol`: augmentations per symbol (default `10`)
- `--flips-per-sample`: bit flips per synthetic sample (default `2`)
- `--vigilance`: vigilance threshold (default `0.82`)
- `--learning-rate`: update rate (default `0.6`)
- `--choice-alpha`: ART choice denominator stabilizer (default `0.001`)

## Data and Model Files

### Pattern images

Pattern discovery expects files that end with `_<LABEL>` and use the following extension:

- `.jpg`

Example:

- `pattern_A.jpg`
- `sample_T.jpg`

### Model files

Trained models are saved as JSON in `patterns_Ass4/`, typically:

- `art1_a_to_t.json`
- `fuzzy_art_a_to_t.json`
- `aug_fuzzy_art_a_to_t.json`

## Notes

- The assignment pipeline assumes symbols **A-T**.
- Augmentation is only applied for `aug_fuzzy_art` during training.
- Dashboard sessions are appended to `terminal.txt` for reproducibility/debugging.
