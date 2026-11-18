import argparse
import sys
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PBL3 = ROOT / "pbl3"
PHOTO_DIR = PBL3 / "photo_models"
sys.path.insert(0, str(PBL3))  # lets joblib load the MixtureOfExperts class

CLASS_ORDER = ["Excellent", "Good", "Fair", "Poor"]
MATERIALS = ["brick", "concrete", "foam", "general_w", "gypsum_board",
             "pipes", "plastic", "stone", "tile", "wood"]
RECOMMENDATION = {
    "Excellent": "Reuse",
    "Good": "Reuse",
    "Fair": "Refurbish or recycle",
    "Poor": "Recycle",
}


def assign_subtype(material):
    """Material subtype using the same rule as the training data conversion."""
    material = material.lower()
    if "wood" in material:
        return "Treated"
    if "concrete" in material:
        return "Reinforced"
    return "Standard"


def read_image(path):
    """Read an image with OpenCV, exiting with a clear message if it fails."""
    file = Path(path)
    if not file.is_file():
        sys.exit(f"Error: file not found: {path}")
    img = cv2.imread(str(file))
    if img is None:
        sys.exit(f"Error: could not read '{path}' as an image (unsupported or corrupt file).")
    return img


def crop_object(img, box):
    """Return (crop, (xmin, ymin, xmax, ymax)); the whole image if no box is given."""
    h, w = img.shape[:2]
    if box is None:
        return img, (0, 0, w, h)
    xmin, ymin, xmax, ymax = box
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)
    if xmax <= xmin or ymax <= ymin:
        sys.exit("Error: --box is empty or outside the image.")
    return img[ymin:ymax, xmin:xmax], (xmin, ymin, xmax, ymax)


def extract_features(img, box=None):
    """Compute the 20 image features exactly as in CODD_Conversion.py.

    Returns (features dict, largest contour or None, box used).
    """
    img_h, img_w = img.shape[:2]
    crop, used_box = crop_object(img, box)
    obj_h, obj_w = crop.shape[:2]
    obj_area = obj_w * obj_h

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 100, 200)

    feats = {
        "pixel_obj_width": obj_w,
        "pixel_obj_height": obj_h,
        "relative_area": obj_area / (img_w * img_h),
        "edge_density": np.sum(edges > 0) / obj_area,
        "color_std": np.std(crop),
        "mean_brightness": np.mean(gray),
        "std_brightness": np.std(gray),
        "contrast": int(np.max(gray)) - int(np.min(gray)),
        "texture_laplacian_var": cv2.Laplacian(gray, cv2.CV_64F).var(),
        "dark_pixel_ratio": np.sum(gray < 50) / obj_area,
        "bright_pixel_ratio": np.sum(gray > 205) / obj_area,
        "mean_hue": np.mean(hsv[:, :, 0]),
        "mean_saturation": np.mean(hsv[:, :, 1]),
        "saturation_std": np.std(hsv[:, :, 1]),
        "contour_area": 0, "contour_perimeter": 0,
        "fill_ratio": 0, "circularity": 0, "solidity": 0, "extent": 0,
    }

    # Largest Otsu-threshold contour, as in training.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        feats["contour_area"] = area
        feats["contour_perimeter"] = perimeter
        feats["fill_ratio"] = area / obj_area
        if perimeter > 0:
            feats["circularity"] = 4 * np.pi * area / (perimeter ** 2)
        hull_area = cv2.contourArea(cv2.convexHull(largest))
        if hull_area > 0:
            feats["solidity"] = area / hull_area
        _, _, bw, bh = cv2.boundingRect(largest)
        if bw * bh > 0:
            feats["extent"] = area / (bw * bh)
    return {k: float(v) for k, v in feats.items()}, largest, used_box


def build_input(feats, material, columns):
    """Assemble the one-row table in the exact training column order."""
    row = dict(feats)
    row.update({
        "Material_Type": material,
        "Material_Subtype": assign_subtype(material),
        "Source": "Demolition_Site",
        "Exposure_Type": "Exterior_Weathered",
    })
    missing = [c for c in columns if c not in row]
    if missing:
        sys.exit(f"Error: cannot build features {missing}; they are not computable here.")
    return pd.DataFrame([row])[columns]


def load_model():
    """Load the saved photo-input stacking model and its training column order."""
    model_file = PHOTO_DIR / "photo_stacking.joblib"
    cols_file = PHOTO_DIR / "photo_feature_columns.joblib"
    if not model_file.exists() or not cols_file.exists():
        sys.exit(f"Error: model files not found in {PHOTO_DIR}.\n"
                 "Create them with: python pbl3/photo_only_comparison.py")
    return joblib.load(model_file), joblib.load(cols_file)


def show_overlay(img, largest, used_box, label):
    """Display the image with the analysed box and largest contour (matplotlib)."""
    import matplotlib.pyplot as plt
    xmin, ymin, xmax, ymax = used_box
    canvas = img.copy()
    cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), max(2, img.shape[1] // 300))
    if largest is not None:
        cv2.drawContours(canvas[ymin:ymax, xmin:xmax], [largest], -1, (0, 0, 255),
                         max(2, img.shape[1] // 400))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {label} (green = analysed region, red = largest contour)")
    plt.axis("off")
    plt.show()


def main():
    """Parse arguments, extract features, predict, and print the report."""
    parser = argparse.ArgumentParser(description="Predict CODD item condition from a photo.")
    parser.add_argument("image", help="path to the image file")
    parser.add_argument("--material", required=True, choices=MATERIALS,
                        help="material of the item (cannot be read from the photo)")
    parser.add_argument("--box", nargs=4, type=int, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="object box in pixels; default is the whole image")
    parser.add_argument("--show", action="store_true", help="show the image with an overlay")
    args = parser.parse_args()

    img = read_image(args.image)
    feats, largest, used_box = extract_features(img, args.box)

    warnings.filterwarnings("ignore")
    model, columns = load_model()
    row = build_input(feats, args.material, columns)

    label = str(model.predict(row)[0])
    proba = dict(zip(model.classes_, model.predict_proba(row)[0]))
    # Generate CSV output
    output = row.copy()

    output["Image"] = args.image
    output["Material"] = args.material
    output["Predicted_Condition"] = label

    for c in CLASS_ORDER:
        output[f"Probability_{c}"] = round(proba.get(c, 0.0) * 100, 2)

    output["Recommendation"] = RECOMMENDATION.get(label, "N/A")

    output_file = "prediction_output.csv"
    output.to_csv(output_file, index=False)

    print(f"\nCSV output saved to: {output_file}")
    print(f"Image: {args.image}")
    print(f"Material (given): {args.material}")
    if args.box is None:
        print("Region: whole image treated as one object (training used annotated object boxes)")
    print("Extracted features:")
    for name in columns:
        print(f"  {name}: {row.iloc[0][name]}")
    print(f"Predicted condition: {label}")
    print("Probabilities: " + ", ".join(
        f"{c} {100 * proba.get(c, 0.0):.1f}%" for c in CLASS_ORDER))
    print(f"Reuse recommendation: {RECOMMENDATION.get(label, 'n/a')}")
    print("Note: Features are estimated from the photo; accuracy may be lower "
          "than the reported test results.")

    if args.show:
        show_overlay(img, largest, used_box, label)


if __name__ == "__main__":
    main()

    