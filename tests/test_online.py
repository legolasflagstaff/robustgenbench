"""End-to-end smoke tests against the Hub (network required): ``pytest -m network``."""
import pytest

import robustgenbench as rgb

pytestmark = pytest.mark.network
DS = "uc-merced-land-use-dataset"


def test_splits_are_aligned():
    test = rgb.load_split(DS, "test")
    adv = rgb.load_split(DS, "linf_30", surrogate="clip_vith14")
    assert len(test) == len(adv) == 420
    assert test["label"] == adv["label"] and test["image_id"] == adv["image_id"]
    assert rgb.class_names(DS)[0] == "agricultural"


def test_every_advertised_split_exists():
    from datasets import get_dataset_config_names, get_dataset_split_names
    configs = set(get_dataset_config_names(rgb.HF_REPO))
    for d in rgb.DATASETS:
        for t in rgb.THREATS:
            for s in rgb.available_surrogates(d, t):
                assert f"{d}-{s}" in configs
                assert t in get_dataset_split_names(rgb.HF_REPO, f"{d}-{s}")


class _Oracle(rgb.BlackBoxClassifier):
    """Answers with the true class name for clean images (via a lookup) and a wrong
    one otherwise, to check scoring, parsing, and caching."""

    def __init__(self):
        self.calls = 0
        self.truth = {}

    def predict(self, image, class_names):
        self.calls += 1
        return self.truth.get(image.tobytes(), "no idea")


def test_blackbox_end_to_end(tmp_path):
    clf = _Oracle()
    test = rgb.load_split(DS, "test").select(range(10))
    names = test.features["label"].names
    clf.truth = {r["image"].tobytes(): names[r["label"]] for r in test}
    res = rgb.evaluate_classifier(clf, "oracle", datasets=[DS], settings=["clean", "linf_30"],
                                  cache_dir=tmp_path, max_examples=10)
    by = {r.setting: r for r in res}
    assert by["clean"].accuracy == 1.0
    assert by["linf_30"].source in rgb.available_surrogates(DS, "linf_30")
    calls = clf.calls
    rgb.evaluate_classifier(clf, "oracle", datasets=[DS], settings=["clean", "linf_30"],
                            cache_dir=tmp_path, max_examples=10)
    assert clf.calls == calls  # second run served from the cache


def test_torch_whitebox_and_transfer():
    torch = pytest.importorskip("torch")
    pytest.importorskip("autoattack")
    from robustgenbench.vision import evaluate_model

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(8), torch.nn.Flatten(),
                                torch.nn.Linear(3 * 64, 21))
    wb = evaluate_model(model, "tiny", datasets=[DS], max_examples=8, batch_size=8, device="cpu")
    assert [r.setting for r in wb] == list(rgb.WHITEBOX_SETTINGS)
    clean = wb[0].accuracy
    assert all(r.accuracy <= clean for r in wb[2:])  # attacks never increase accuracy
    tr = evaluate_model(model, "tiny", datasets=[DS], mode="transfer",
                        settings=["clean", "linf_30"], max_examples=8, batch_size=8,
                        device="cpu")
    assert tr[1].source in rgb.available_surrogates(DS, "linf_30")
