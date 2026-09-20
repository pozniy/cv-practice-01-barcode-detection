import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectorConfig:

    working_width: int = 1000
    blackhat_kernel: tuple[int, int] = (15, 5)
    blur_kernel: tuple[int, int] = (9, 9)
    morphology_kernel: tuple[int, int] = (21, 5)
    erode_iterations: int = 2
    dilate_iterations: int = 4

    min_area_fraction: float = 0.003
    max_area_fraction: float = 0.40
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 8.0
    min_width: int = 40
    min_height: int = 15


DEFAULT_CONFIG = DetectorConfig()


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:

    ax, ay, aw, ah = (float(value) for value in box_a)
    bx, by, bw, bh = (float(value) for value in box_b)

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _resize_to_working_width(
    image: np.ndarray, working_width: int
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = working_width / width
    if width == working_width:
        return image.copy(), 1.0
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (working_width, resized_height), interpolation=interpolation)
    return resized, scale


def build_feature_maps(
    image: np.ndarray, config: DetectorConfig = DEFAULT_CONFIG
) -> dict[str, Any]:

    if image is None or image.size == 0:
        raise ValueError("The input image is empty")

    working, scale = _resize_to_working_width(image, config.working_width)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    blackhat_element = cv2.getStructuringElement(
        cv2.MORPH_RECT, config.blackhat_kernel
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_element)


    gradient_float = np.abs(cv2.Scharr(blackhat, cv2.CV_32F, 1, 0))
    gradient = cv2.normalize(
        gradient_float, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    blurred = cv2.GaussianBlur(gradient, config.blur_kernel, 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    gray_gradient_x = np.abs(cv2.Scharr(gray, cv2.CV_32F, 1, 0))
    gray_gradient_y = np.abs(cv2.Scharr(gray, cv2.CV_32F, 0, 1))

    morphology_mask = _make_morphology_mask(binary, config)

    return {
        "working": working,
        "scale": scale,
        "gray": gray,
        "blackhat": blackhat,
        "gradient": gradient,
        "blurred": blurred,
        "binary": binary,
        "gradient_x": gray_gradient_x,
        "gradient_y": gray_gradient_y,
        "morphology_mask": morphology_mask,
    }


def _candidate_score(box: BBox, maps: dict[str, Any]) -> float:
    x, y, width, height = box
    roi_x = maps["gradient_x"][y : y + height, x : x + width]
    roi_y = maps["gradient_y"][y : y + height, x : x + width]

    mean_x = float(roi_x.mean())
    mean_y = float(roi_y.mean())
    return mean_x / (mean_x + mean_y + 1e-6)


def _working_to_original_box(box: BBox, scale: float, shape: tuple[int, ...]) -> BBox:
    x, y, width, height = box
    original_height, original_width = shape[:2]
    x1 = int(np.clip(round(x / scale), 0, original_width - 1))
    y1 = int(np.clip(round(y / scale), 0, original_height - 1))
    x2 = int(np.clip(round((x + width) / scale), x1 + 1, original_width))
    y2 = int(np.clip(round((y + height) / scale), y1 + 1, original_height))
    return x1, y1, x2 - x1, y2 - y1


def _make_morphology_mask(
    binary: np.ndarray, config: DetectorConfig
) -> np.ndarray:
    element = cv2.getStructuringElement(cv2.MORPH_RECT, config.morphology_kernel)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    mask = cv2.erode(mask, None, iterations=config.erode_iterations)
    return cv2.dilate(mask, None, iterations=config.dilate_iterations)


def detect_barcode(
    image: np.ndarray,
    config: DetectorConfig = DEFAULT_CONFIG,
    return_debug: bool = False,
) -> tuple[BBox | None, dict[str, Any] | None]:
    maps = build_feature_maps(image, config)
    image_height, image_width = maps["gray"].shape
    candidates: list[dict[str, Any]] = []

    mask = maps["morphology_mask"]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        box = (x, y, width, height)
        area_fraction = width * height / (image_width * image_height)
        aspect_ratio = width / max(height, 1)

        if not (
            config.min_area_fraction <= area_fraction <= config.max_area_fraction
            and config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio
            and width >= config.min_width
            and height >= config.min_height
        ):
            continue

        candidates.append(
            {
                "box": _working_to_original_box(box, maps["scale"], image.shape),
                "score": _candidate_score(box, maps),
            }
        )

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    prediction = candidates[0]["box"] if candidates else None

    if not return_debug:
        return prediction, None

    debug = dict(maps)
    debug["candidates"] = candidates
    return prediction, debug


def draw_detection(
    image: np.ndarray,
    prediction: BBox | None,
    ground_truth: BBox | None = None,
    iou: float | None = None,
) -> np.ndarray:

    result = image.copy()
    if ground_truth is not None:
        x, y, width, height = ground_truth
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 0, 255), 3)
    if prediction is not None:
        x, y, width, height = prediction
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 3)
    if iou is not None:
        cv2.putText(
            result,
            f"IoU: {iou:.3f}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
    return result


def evaluate_dataset(
    data_dir: str | Path,
    config: DetectorConfig = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
    save_images: bool = False,
) -> pd.DataFrame:

    data_dir = Path(data_dir)
    annotations_path = data_dir / "annotations.tsv"
    annotations = pd.read_csv(annotations_path, sep=",")
    required = {"filename", "x_from", "y_from", "width", "height"}
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")

    target_dir = Path(output_dir) if output_dir is not None else None
    visualization_dir = target_dir / "visualizations" if target_dir else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)
    if save_images and visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for annotation in annotations.itertuples(index=False):
        image_path = data_dir / annotation.filename
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        prediction, debug = detect_barcode(image, config, return_debug=True)
        ground_truth = (
            int(annotation.x_from),
            int(annotation.y_from),
            int(annotation.width),
            int(annotation.height),
        )
        value = box_iou(prediction, ground_truth) if prediction is not None else 0.0
        score = debug["candidates"][0]["score"] if debug["candidates"] else 0.0

        if prediction is None:
            px = py = pw = ph = -1
        else:
            px, py, pw, ph = prediction
        rows.append(
            {
                "filename": annotation.filename,
                "x_from": px,
                "y_from": py,
                "width": pw,
                "height": ph,
                "score": score,
                "iou": value,
            }
        )

        if save_images and visualization_dir is not None:
            rendered = draw_detection(image, prediction, ground_truth, value)
            cv2.imwrite(str(visualization_dir / Path(annotation.filename).name), rendered)

    results = pd.DataFrame(rows)
    if target_dir is not None:
        results.to_csv(target_dir / "predictions.csv", index=False)
        metrics = summarize_results(results)
        (target_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return results


def summarize_results(results: pd.DataFrame) -> dict[str, float | int]:

    return {
        "images": int(len(results)),
        "mean_iou": float(results["iou"].mean()),
        "median_iou": float(results["iou"].median()),
        "min_iou": float(results["iou"].min()),
        "images_iou_at_least_0_8": int((results["iou"] >= 0.8).sum()),
        "fraction_iou_at_least_0_8": float((results["iou"] >= 0.8).mean()),
    }


def _print_metrics(metrics: dict[str, float | int]) -> None:
    print(f"Images:             {metrics['images']}")
    print(f"Mean IoU:           {metrics['mean_iou']:.4f}")
    print(f"Median IoU:         {metrics['median_iou']:.4f}")
    print(f"Minimum IoU:        {metrics['min_iou']:.4f}")
    print(
        "IoU >= 0.8:         "
        f"{metrics['images_iou_at_least_0_8']}/{metrics['images']} "
        f"({metrics['fraction_iou_at_least_0_8']:.1%})"
    )


def _run_single_image(args: argparse.Namespace) -> None:
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    prediction, debug = detect_barcode(image, return_debug=True)
    score = debug["candidates"][0]["score"] if debug["candidates"] else 0.0
    print(json.dumps({"bbox": prediction, "score": score}, ensure_ascii=False))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rendered = draw_detection(image, prediction)
        cv2.imwrite(str(args.output_dir / Path(args.image).name), rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/classicData"),
        help="Directory with annotations.tsv and images/",
    )
    parser.add_argument("--image", type=Path, help="Detect a single image instead")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/barcode_detection"),
        help="Where predictions, metrics and optional images are saved",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save images containing prediction and ground-truth boxes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image is not None:
        _run_single_image(args)
        return

    results = evaluate_dataset(
        args.data_dir,
        output_dir=args.output_dir,
        save_images=args.save_images,
    )
    _print_metrics(summarize_results(results))
    print(f"Results saved to:   {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
