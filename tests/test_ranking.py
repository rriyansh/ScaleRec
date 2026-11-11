import os


def test_ranker_saved(prepare_artifacts):
    assert os.path.exists("models_store/xgb_ranker.json")
    assert os.path.getsize("models_store/xgb_ranker.json") > 0
