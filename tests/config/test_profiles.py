import pytest
from ragnarok.config.schema import AppConfig
from ragnarok.config.profiles import ProfileStore


def test_save_list_load_roundtrip(tmp_path):
    store = ProfileStore(tmp_path / "profiles")
    assert store.list() == []
    cfg = AppConfig().model_copy(
        update={"aim": AppConfig().aim.model_copy(update={"kp": 0.77})})
    store.save("AK-47", cfg)
    assert store.list() == ["AK-47"] and store.exists("AK-47")
    loaded = store.load("AK-47")
    assert loaded.aim.kp == 0.77


def test_list_is_sorted_and_delete_removes(tmp_path):
    store = ProfileStore(tmp_path)
    store.save("zebra", AppConfig())
    store.save("alpha", AppConfig())
    assert store.list() == ["alpha", "zebra"]
    store.delete("alpha")
    assert store.list() == ["zebra"]
    store.delete("missing")                      # deleting a non-existent name is a no-op


def test_load_missing_raises_not_autocreate(tmp_path):
    store = ProfileStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("nope")
    assert store.list() == []                    # load must NOT have created a file


def test_invalid_names_rejected(tmp_path):
    store = ProfileStore(tmp_path)
    for bad in ("", "..", "a/b", "a\\b", "we/../etc"):
        with pytest.raises(ValueError):
            store.path_for(bad)


def test_list_empty_when_dir_absent(tmp_path):
    assert ProfileStore(tmp_path / "does_not_exist").list() == []
