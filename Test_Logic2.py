"""Functional tests for the core logic.   python -m pytest -q test_logic.py   (or: python test_logic.py)"""
import numpy as np, cv2
from unittest.mock import patch, MagicMock
from pipeline.recommend import recommend
from pipeline.features import extract_features, FEATURE_NAMES
from pipeline.detect import Detector, generate_explanation

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


# ------------------------------------------------------------------ Claude LLM integration
# All mocked: these must pass with zero network calls and no ANTHROPIC_API_KEY set, so
# the suite stays fast, free, and deterministic in CI and in a live demo environment.

def test_claude_assess_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    det = Detector()
    assert det.claude is None
    assert det.claude_assess(np.full((50, 50, 3), 120, np.uint8)) is None

def test_claude_assess_parses_a_well_formed_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    fake_reply = MagicMock()
    fake_reply.content = [MagicMock(text='{"material": "brick", "damage_description": '
                                          '"minor surface cracking", "damage_severity": 0.3}')]
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = fake_reply
        det = Detector()
        result = det.claude_assess(np.full((50, 50, 3), 120, np.uint8))
    assert result["material"] == "brick"
    assert result["damage_severity"] == 0.3
    assert result["source"] == "claude-vision"

def test_claude_assess_never_crashes_the_scan_on_api_failure(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.side_effect = RuntimeError("network down")
        det = Detector()
        result = det.claude_assess(np.full((50, 50, 3), 120, np.uint8))
    assert "error" in result and result["source"] == "claude-vision"   # degrades, never raises

def test_claude_assess_tolerates_a_stray_preamble(monkeypatch):
    # models occasionally wrap JSON in prose despite instructions; must still parse
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    fake_reply = MagicMock()
    fake_reply.content = [MagicMock(text='Sure, here you go:\n{"material": "concrete", '
                                          '"damage_description": "none", "damage_severity": 0.0}')]
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = fake_reply
        det = Detector()
        result = det.claude_assess(np.full((50, 50, 3), 120, np.uint8))
    assert result["material"] == "concrete"

def test_explanation_falls_back_to_template_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rec = {"pathway": "REFURBISH", "alternative_uses": ["landscaping"]}
    text = generate_explanation("brick", 62, rec)
    assert "brick" in text.lower() and "REFURBISH" in text and "62" in text

def test_explanation_never_contradicts_the_decided_pathway(monkeypatch):
    # the LLM narrates; it must not be able to silently change the pathway it was given
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    fake_reply = MagicMock()
    fake_reply.content = [MagicMock(text="This brick scored 62/100 and is recommended "
                                          "for REFURBISH, such as landscaping use.")]
    rec = {"pathway": "REFURBISH", "alternative_uses": ["landscaping"]}
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = fake_reply
        text = generate_explanation("brick", 62, rec)
    assert "REFURBISH" in text
    assert "RECYCLE" not in text and "REUSE" not in text

def test_explanation_falls_back_to_template_on_api_failure(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    rec = {"pathway": "RECYCLE", "alternative_uses": []}
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.side_effect = RuntimeError("timeout")
        text = generate_explanation("gypsum", 15, rec)
    assert "gypsum" in text.lower() and "RECYCLE" in text   # template fallback, still correct


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    import inspect
    for t in tests:
        if "monkeypatch" in inspect.signature(t).parameters:
            print("SKIP (needs pytest fixture, run via `python -m pytest`)", t.__name__)
            continue
        t(); print("PASS", t.__name__)
    print(f"done")
