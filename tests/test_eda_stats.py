"""eda_stats.py helpers: histogram binning, OpenVid source parsing, caption tokenising, weather flags, cuts."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eda_stats", ROOT / "scripts" / "data" / "eda_stats.py")
eda = importlib.util.module_from_spec(spec); spec.loader.exec_module(eda)  # type: ignore[union-attr]


def test_num_stats_hist_accounts_for_every_value():
    v = [0.0, 1.0, 1.5, 2.0, 3.0, 10.0, -5.0, None, float("nan")]
    s = eda.num_stats(v, edges=[0, 1, 2, 3])
    h = s["hist"]
    assert s["n"] == 7 and s["missing"] == 2
    assert h["counts"] == [1, 2, 2]          # [0,1) [1,2) [2,3] (numpy: last bin closed)
    assert h["below"] == 1 and h["above"] == 1
    assert sum(h["counts"]) + h["below"] + h["above"] == s["n"]
    assert s["min"] == -5 and s["max"] == 10 and s["median"] == 1.5


def test_num_stats_auto_edges_and_constant_input():
    s = eda.num_stats(np.arange(1000), bins=10)
    assert len(s["hist"]["edges"]) == 11 and sum(s["hist"]["counts"]) + s["hist"]["below"] + s["hist"]["above"] == 1000
    c = eda.num_stats([4.0] * 5)
    assert sum(c["hist"]["counts"]) == 5 and c["hist"]["below"] == c["hist"]["above"] == 0
    assert eda.num_stats([]) == {"n": 0, "missing": 0}


def test_cat_sorts_and_truncates():
    c = eda.cat({"a": 1, "b": 5, "c": 3}, top=2)
    assert c["labels"] == ["b", "c"] and c["counts"] == [5, 3] and c["other"] == 1 and c["n"] == 9


def test_source_dataset_from_names():
    names = {
        "data/v/OpenVid_part1/celebv_--Jiv5iYqT8_0.mp4": "celebv",
        "ECfF08Vn_5w_14_0to274.mp4": "panda",
        "magictime_vGOR2_vlq9M_0001.mp4": "magictime",
        "magictime_10rMXQ6k0Y.mp4": "magictime",          # 10-char id: unparsed by video_source, prefix fallback
        "pixabay_Spring_Blossoms_39162_002.mp4": "pixabay",
        "abcdefghij_weird.mp4": "unparsed",              # lowercase start is not a known prefix
    }
    assert {n: eda.source_dataset(n) for n in names} == names


def test_caption_tokens_and_mentions():
    toks = eda.caption_tokens("The video features a Man's face; he's smiling. 3 dogs!")
    assert toks == ["the", "video", "features", "a", "man's", "face", "he's", "smiling", "dogs"]
    assert eda.mentions(toks, eda.FACE_WORDS)
    assert eda.mentions(eda.caption_tokens("A woman walks"), eda.PERSON_WORDS)
    assert not eda.mentions(eda.caption_tokens("A train passes the terrain"), eda.PERSON_WORDS)
    assert "the" in eda.STOPWORDS and "video" not in eda.STOPWORDS


def test_weather_flags_word_boundaries():
    assert eda.weather_flags("Night, rain, peds") == {"night": True, "rain": True}
    assert eda.weather_flags("Rainy day at the train station") == {"night": False, "rain": True}
    assert eda.weather_flags("Wait at intersection, drain, terrain, nightclub") == {"night": False, "rain": False}


def test_pixel_helpers():
    gray = np.full((8, 8, 3), 120.0)
    assert eda.colourfulness(gray) == 0.0 and eda.laplacian_var(gray[..., 0]) == 0.0
    red = np.zeros((8, 8, 3)); red[..., 0] = 255
    assert eda.colourfulness(red) > 50


def test_hard_cuts_ratio_and_floor():
    d = [0.5, 0.6, 0.5, 30.0, 0.55]
    assert eda.hard_cuts(d) == 1 and eda.hard_cuts(d, floor=10) == 1
    static = [0.01, 0.01, 0.2, 0.01]                    # median ~0: ratio alone flags noise, the floor does not
    assert eda.hard_cuts(static) == 1 and eda.hard_cuts(static, floor=10) == 0


def test_flatten_criteria_numbers_lists_and_categoricals():
    crit = {"ego_disp_m": 3.5, "ok": True, "moving": [{"instance": "abc", "category": "vehicle.car", "disp_m": 4.0}],
            "vis": [4, 3], "variant": "lidar_dropped", "partner_id": "x"}
    got = sorted(eda.flatten_criteria(crit))
    assert got == sorted([("ego_disp_m", 3.5), ("moving[].category", "vehicle.car"), ("moving[].disp_m", 4.0),
                          ("vis[]", 4), ("vis[]", 3), ("variant", "lidar_dropped")])


def test_angle_helpers():
    assert eda.wrap_deg(350) == -10 and eda.wrap_deg(-190) == 170
    assert abs(eda.quat_yaw((np.cos(0.5), 0, 0, np.sin(0.5))) - 1.0) < 1e-12
