# RecPatGen — Recursive Pattern Generator

RecPatGen is a desktop application (Python + Tkinter) that transforms a source image into abstract, symmetry-driven generative art. You chain together a pipeline of image operations, adjust each one's parameters, preview the output next to the original, then save a single result or batch-generate many random variations.

## What it does

1. Load an image (JPG, PNG, BMP, TIFF, WebP). It is converted to RGB and downscaled so its longest side is at most the configured **Max dim** (default 1500 px).
2. Add operations to a **pipeline**. Each run starts from the original image and applies the operations in order, top to bottom.
3. View the result beside the original, with a progress bar and ETA.
4. Save the result (PNG/JPEG), or run a **batch** that repeats the pipeline N times and writes numbered PNGs. Because operations are randomised, every batch image differs.

### Operations

| Operation | Effect |
|---|---|
| Fractal Mirroring | Recursively applies symmetries to nested sub-squares across the image |
| Colour Region Symmetry | Finds regions by a selection method (brightness, RGB channel, edges, HSV, texture, k-means segmentation, entropy, random…) and applies symmetry to patches centred on them |
| Vertical / Horizontal Structures | Tiles symmetrised strips as columns or rows, with optional perpendicular extensions |
| Pattern Cascade | Wavy, blended trails of symmetrised patterns running down the image |
| Diamond / Geo Patterns | Nested shapes (diamond, circle, cross, X, oval, line) with turbulence warping and feathered edges |
| Pixel Sorting | Sorts pixels in random regions by brightness, red, green or hue |
| Geometric Grid | Grid cells filled with layered shapes using locally sampled colours |
| Recursive Glitch | Band shifts, selective mirroring and pattern duplication |
| Vertical Cascade | Drip/smear effect with bleed, sine-wave drift, plasma colour and branching |
| Fractal Draw | Recursive coloured subdivision + iso-contour lines + an IFS dot cloud |
| Grid Transform | Applies one (or a random) operation per grid cell: symmetry, mirror, rotate, invert, shuffle, channel shift, pixel sort, etc. |
| Mirror Sections | Mirrors the top half of horizontal bands onto the bottom half |
| Blend With Original | Mixes the current result with the source image |
| Final Effects | Contrast, sharpen, optional vignette |

Available symmetries: vertical, horizontal, quad, radial, kaleidoscope3, kaleidoscope6, glide, rotate90, rotate180, wave, spiral, fractal, wallpaper, reflective_tiling.

## Requirements

- Python 3.8 or newer, with Tkinter
- Packages listed in `requirements.txt`: NumPy, Pillow, SciPy, scikit-learn (1.2+), opencv-python
- A desktop environment (it is a GUI app; it will not run headless)

## Installation

```bash
# 1. Get the script. The source was supplied as RecPatGenFinal2.txt - rename it:
mv RecPatGenFinal2.txt RecPatGenFinal2.py

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# Linux only, if Tkinter is missing:
sudo apt install python3-tk
```

## Running

```bash
python RecPatGenFinal2.py
```

Basic workflow:

1. **01 — Source Image:** set *Max dim* (optional), click **OPEN IMAGE**.
2. **02 — Pipeline:** choose an operation from the dropdown, click **+ ADD**. Repeat to chain more. Use the ↑ / ↓ / ✕ buttons on each card to reorder or remove.
3. **03 — Actions:** **RUN PIPELINE** to generate, **SAVE RESULT** to export (PNG or JPEG), **RESET IMAGE** to clear the result.
4. **04 — Batch Generate:** set *Count*, filename *Prefix* and an output *Folder* (via **…**), then **RUN BATCH**. Files are saved as `<prefix>_0001.png`, `<prefix>_0002.png`, … Use **STOP** to cancel between images.

## Configuration

There are no config files or command-line options; everything is set in the GUI.

- **Max dim** — longest side in pixels the image is scaled to on load (minimum 100). Lower it for faster previews.
- **Operation parameters** — each pipeline card has a slider and an editable number box per parameter. You can type values above the slider's maximum (only the minimum is enforced). Percent parameters are converted to fractions internally.
- **Symmetry presets** — dropdowns offer groups (e.g. "Quad + Radial", "Kaleidoscope 3+6", "All Types") or a single specific symmetry.
- **Mirror Sections → Section H** — 0 means one third of the image height.
- **Appearance and defaults** — colours, fonts and default values are constants at the top of the script and in the `OPS` table, and can be edited in the source.
- **Logging** — INFO-level messages are printed to the console.

## Limitations and known issues

- **Resolution:** output is at the downscaled working resolution (≤ Max dim), not the original file's size. DPI metadata is preserved; transparency is discarded (image converted to RGB).
- **Randomness:** results are non-deterministic and there is no seed control, so a given result cannot be reproduced exactly (except by saving it).
- **Performance:** several operations use Python-level loops and can be slow on large images or high counts — notably Vertical Cascade, Fractal Draw (IFS cloud), Pixel Sorting (hue), Geometric Grid, and the Texture/Entropy selection methods in Colour Region Symmetry. The pipeline cannot be cancelled mid-run (only batches can be stopped, and only between operations/images).
- **Colour Region Symmetry:** selection-method parameters (thresholds, colour ranges, HSV ranges, cluster count) are not exposed in the GUI and use their defaults, so *color_range* and *hsv* select the whole image by default. *Size Mode* should lie between *Size Min* and *Size Max* or the size distribution is invalid.
- **Pipeline card buttons:** the ↑ / ↓ / ✕ callbacks are bound to the card's original position, so after removing or reordering cards the buttons on other cards may act on the wrong item. If this happens, re-add the operations in the desired order.
- **Vignette:** the vignette option has little or no visible effect (the alpha mask is dropped when converting back to RGB).
- **Batch mode:** saves PNG only, overwrites existing files with the same names, does not update the preview, and shares the same pipeline settings for all images.
- **No undo, no saving/loading of pipelines or presets.**
- **Symmetry quirks:** kaleidoscope/spiral modes combine rotations with a per-pixel maximum, which tends to brighten results; the "wave" symmetry uses a fixed 10 px displacement.
