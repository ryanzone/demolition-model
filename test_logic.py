"""Functional tests for the core logic.   python -m pytest -q test_logic.py   (or: python test_logic.py)"""
import numpy as np, cv2
from pipeline.recommend import recommend
from pipeline.features import extract_features, FEATURE_NAMES

CLEAN = dict.fromkeys(FEATURE_NAMES, 0.0)

def f(**kw): return {**CLEAN, **kw}

def test_good_material_is_reused():
    assert recommend("brick", 85, f())["pathway"] == "REUSE"

def test_mid_score_is_refurbished():
    assert recommend("brick", 55, f())["pathway"] == "REFURBISH"

def test_low_score_is_recycled():
    assert recommend("brick", 20, f())["pathway"] == "RECYCLE"

def test_heavy_cracking_blocks_direct_reuse():
    assert recommend("brick", 90, f(crack_area_ratio=0.15))["pathway"] == "REFURBISH"

def test_gypsum_never_reused():
    assert recommend("gypsum", 99, f())["pathway"] != "REUSE"

def test_unknown_material_uses_default_rules():
    r = recommend("unknown", 80, f())
    assert r["pathway"] == "REUSE" and r["alternative_uses"]

def test_every_result_carries_disclaimer():
    assert "not a structural certification" in recommend("wood", 50, f())["disclaimer"]

def test_thresholds_are_monotonic():          # a better score never gives a worse pathway
    order = {"RECYCLE": 0, "REFURBISH": 1, "REUSE": 2}
    for m in ("brick", "concrete", "tile", "wood", "gypsum", "plastic", "unknown"):
        seq = [order[recommend(m, s, f())["pathway"]] for s in range(0, 101, 5)]
        assert seq == sorted(seq), m

def test_features_on_undamaged_image():
    img = np.full((200, 300, 3), 130, np.uint8)
    ft = extract_features(img, [], np.zeros((200, 300), np.uint8))
    assert list(ft) == FEATURE_NAMES and ft["crack_area_ratio"] == 0 and ft["n_damage_regions"] == 0

def test_features_measure_damage_inside_detection_box_only():
    img = np.full((200, 300, 3), 130, np.uint8)
    mask = np.zeros((200, 300), np.uint8); mask[10:20, 10:290] = 1       # crack OUTSIDE the box
    box = [{"label": "brick", "confidence": 0.9, "box": [0, 100, 300, 200]}]
    assert extract_features(img, box, mask)["crack_area_ratio"] == 0
    mask[150:155, 20:280] = 1                                            # crack INSIDE the box
    assert extract_features(img, box, mask)["crack_area_ratio"] > 0.02

def test_api_rejects_bad_input_and_returns_report():
    from fastapi.testclient import TestClient
    from main import app
    c = TestClient(app)
    assert c.post("/api/assess", files={"file": ("x.txt", b"nope", "text/plain")}).status_code == 400
    assert c.post("/api/assess", files={"file": ("big.jpg", b"0" * (11 * 2**20), "image/jpeg")}).status_code == 413
    ok, buf = cv2.imencode(".jpg", np.full((300, 400, 3), 140, np.uint8))
    j = c.post("/api/assess", files={"file": ("a.jpg", buf.tobytes(), "image/jpeg")}).json()
    assert j["recommendation"]["pathway"] in {"REUSE", "REFURBISH", "RECYCLE"} and 0 <= j["condition"]["score"] <= 100
    assert c.get(f"/api/reports/{j['id']}").status_code == 200 and c.get("/api/reports/nope").status_code == 404

if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests: t(); print("PASS", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
