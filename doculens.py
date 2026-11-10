from pathlib import Path
import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """Return four points in top-left, top-right, bottom-right, bottom-left order."""
    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)], pts[np.argmin(diff)],
        pts[np.argmax(s)], pts[np.argmax(diff)]
    ], dtype="float32")


def find_document_contour(image: np.ndarray):
    """Find the largest plausible four-corner contour; return None if not found."""
    h, w = image.shape[:2]
    scale = 900.0 / max(h, w) if max(h, w) > 900 else 1.0
    small = cv2.resize(image, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.08 * small.shape[0] * small.shape[1]:
            return approx.reshape(4, 2).astype("float32") / scale
    return None


def four_point_warp(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    tl, tr, br, bl = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width, height = max(width, 1), max(height, 1)
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def enhance_document(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    threshold = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 12
    )
    blur = cv2.GaussianBlur(denoised, (0, 0), 1.2)
    sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
    return gray, threshold, sharpened


def process(image_path, output_dir, warp=True):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    edges = cv2.Canny(
        cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0),
        50, 150
    )
    cv2.imwrite(str(output_dir / f"{image_path.stem}_edges.jpg"), edges)

    working = image
    if warp:
        corners = find_document_contour(image)
        if corners is not None:
            working = four_point_warp(image, corners)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_warped.jpg"), working)
        else:
            print("No confident four-corner page contour found; enhancing original image.")

    gray, threshold, enhanced = enhance_document(working)
    cv2.imwrite(str(output_dir / f"{image_path.stem}_gray.jpg"), gray)
    cv2.imwrite(str(output_dir / f"{image_path.stem}_threshold.jpg"), threshold)
    cv2.imwrite(str(output_dir / f"{image_path.stem}_enhanced.jpg"), enhanced)
    return {"input_shape": image.shape, "processed_shape": working.shape,
            "output_dir": str(output_dir)}
