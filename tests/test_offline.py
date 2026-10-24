"""Tests that run without network access."""
import pandas as pd
import pytest

import robustgenbench as rgb


def test_reference_reproduces_paper_leaderboard():
    ref = rgb.reference("vision_whitebox", protocol="fft50")
    assert ref.model.nunique() == 80
    s, b = rgb.sum_score(ref), rgb.borda_score(ref)
    top = "convnext_b,sup,in22k | TRADES"
    assert s.index[0] == top and b.index[0] == top          # Table 1 of the paper
    assert s[top] == pytest.approx(19.789, abs=1e-3)
    assert b[top] == 2281                                    # max Borda in Figure 4
    assert b.min() == 329                                    # min Borda in Figure 4
    assert rgb.kendall_tau(b, s) == pytest.approx(0.854, abs=1e-3)


def test_mac_on_reference():
    ref = rgb.reference("vision_whitebox", protocol="fft50")
    assert rgb.mean_absolute_correlation(ref) == pytest.approx(0.795, abs=2e-3)


def test_vellm_reference_worst_case():
    ref = rgb.worst_case_over_surrogates(rgb.reference("vellm_transfer"))
    gem = ref[ref.model == "gemini-3-flash (no think)"]
    assert gem.groupby("setting").size().eq(6).all()
    clean = gem[gem.setting == "clean"].accuracy.mean()
    assert clean == pytest.approx(0.918, abs=1e-3)


@pytest.mark.parametrize("raw,expected", [
    ("Sunflower", 1), ("sunflower.", 1), ("The answer is: sunflower", 1),
    ("pink primrose", 0), ("pink primrise", 0), ("airplane", None)])
def test_parse_prediction(raw, expected):
    assert rgb.parse_prediction(raw, ["pink primrose", "sunflower", "rose"]) == expected


def test_prompt_matches_paper():
    system, user = rgb.build_prompt(["a", "b"])
    assert system.startswith("Classify the image into exactly one of these classes: a, b.")
    assert user == "Classify this image."


def test_coverage():
    assert rgb.available_surrogates("stanford_cars", "linf_4") == [
        "clip_vitb16", "clip_vith14", "metaclip_vith14"]
    assert "siglip2_base" in rgb.available_surrogates("caltech101", "linf_30")
    assert "siglip2_so400m" in rgb.available_surrogates("caltech101", "l1_75")


def test_leaderboard_filters_to_evaluated_pairs():
    new = pd.DataFrame({"model": "mine", "dataset": "flowers-102",
                        "setting": ["clean", "common"], "accuracy": [1.0, 1.0]})
    table = rgb.leaderboard(new, compare_with="fft50")
    assert table.index[0] == "mine" and table.sum_score.iloc[0] == 2.0
    assert len(table) == 81
