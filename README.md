# DocuLens — Document Image Processing

DocuLens is a classical computer-vision pipeline for document photographs. It combines edge detection, contour-based page detection, perspective correction and image enhancement.

## Features
- Gaussian smoothing and Canny edge detection
- Four-corner document contour detection
- Perspective correction using a four-point transform
- Bilateral denoising, adaptive thresholding and sharpening
- Graceful fallback when a page boundary cannot be detected

## Setup
```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python run.py --image path/to/document.jpg --output outputs
```

To skip perspective correction:
```bash
python run.py --image path/to/document.jpg --output outputs --no-warp
```

## Output files
The output directory contains an edge map, grayscale image, adaptive-threshold image and enhanced image. A perspective-corrected image is also saved when a confident page contour is detected.

## Limitations
Contour-based page detection depends on lighting, contrast and visible page boundaries. Low-contrast images or cluttered backgrounds may prevent reliable corner detection; in those cases the pipeline enhances the original image instead.

## Project status
Implemented and tested on a small set of document photographs. Detection is not guaranteed for every input; inspect the output before using it for archival or OCR workflows.
