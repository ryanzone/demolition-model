"""Build sample predict.py commands from HELD-OUT test items only.

For every material it picks 3 random items from testing_image_features_grouped.csv
(the test split the models never saw), finds the matching photo and box, runs the
photo-input model, and writes:
  sample_queries.txt        one ready-to-paste predict.py command per sample (with --show)
  sample_results.csv        material, image, box, true label, predicted label, probability
  sample_outputs/*.png      overlay image for each sample

Run:  python make_sample_queries.py
"""
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

import predict  # reuses the exact feature extraction and model from predict.py

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "CODD" / "Construction and Demolition Waste Object Detection Dataset  (CODD)"
SPLIT_FOLDERS = ["training", "validation", "testing"]
PER_MATERIAL = 3
OUT_DIR = ROOT / "sample_outputs"


def find_source(row):
    """Find the photo and box that produced this CSV row (verified by edge_density)."""
    name = str(row.Image_ID)
    for folder in SPLIT_FOLDERS:
        xml_path = DATASET / folder / name.replace(".jpg", ".xml")
        img_path = DATASET / folder / name
        if not xml_path.exists() or not img_path.exists():
            continue
        for obj in ET.parse(xml_path).getroot().findall("object"):
            if obj.find("name").text != row.Material_Type:
                continue
            b = obj.find("bndbox")
            box = [int(float(b.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]
            if box[2] - box[0] != row.pixel_obj_width or box[3] - box[1] != row.pixel_obj_height:
                continue
            img = predict.read_image(str(img_path))
            feats, largest, used = predict.extract_features(img, box)
            if abs(feats["edge_density"] - row.edge_density) < 1e-6:   # same image confirmed
                return img_path, box, img, largest, used
    return None


def main():
    """Pick held-out samples, predict them, and write the command list and outputs."""
    test = pd.read_csv(ROOT / "testing_image_features_grouped.csv")
    model, columns = predict.load_model()
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.RandomState(42)

    rows, lines = [], []
    for material in predict.MATERIALS:
        pool = test[test.Material_Type == material]
        picked = 0
        for idx in rng.permutation(len(pool)):
            if picked >= PER_MATERIAL:
                break
            row = pool.iloc[idx]
            found = find_source(row)
            if found is None:
                continue
            img_path, box, img, largest, used = found
            feats, _, _ = predict.extract_features(img, box)
            x = predict.build_input(feats, material, columns)
            label = str(model.predict(x)[0])
            proba = dict(zip(model.classes_, model.predict_proba(x)[0]))
            picked += 1

            rel = img_path.relative_to(ROOT).as_posix()
            cmd = f'python predict.py "{rel}" --material {material} --box {" ".join(map(str, box))} --show'
            lines += [f"# {material} | TRUE: {row.Condition_Class} | PREDICTED: {label} "
                      f"({100 * proba[label]:.1f}%)", cmd, ""]
            png = OUT_DIR / f"{material}_{picked}_true-{row.Condition_Class}_pred-{label}.png"
            predict.show_overlay(img, largest, used, label, 100 * proba[label], material,
                                 save_path=str(png), show=False)
            rows.append({"material": material, "image": rel, "box": " ".join(map(str, box)),
                         "true_label": row.Condition_Class, "predicted": label,
                         "probability_%": round(100 * proba[label], 1),
                         "correct": row.Condition_Class == label})

    (ROOT / "sample_queries.txt").write_text("\n".join(lines), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "sample_results.csv", index=False)
    print(f"{len(df)} held-out samples, {int(df.correct.sum())} predicted correctly "
          f"({100 * df.correct.mean():.0f}%)")
    print(df[["material", "true_label", "predicted", "probability_%"]].to_string(index=False))


if __name__ == "__main__":
    main()
