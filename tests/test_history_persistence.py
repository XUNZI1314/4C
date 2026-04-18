import json
from pathlib import Path

from protein_visualizer.services import session_state


def _install_fake_path_io(monkeypatch):
    storage: dict[str, str] = {}
    directories: set[str] = set()

    def _mkdir(self, parents=False, exist_ok=False):
        directories.add(str(self))

    def _write_text(self, text, encoding=None):
        storage[str(self)] = str(text)
        return len(str(text))

    def _read_text(self, encoding=None):
        return storage[str(self)]

    def _exists(self):
        return str(self) in storage or str(self) in directories

    def _replace(self, target):
        storage[str(target)] = storage.pop(str(self))
        return Path(target)

    def _unlink(self, missing_ok=False):
        key = str(self)
        if key in storage:
            storage.pop(key, None)
            return None
        if not missing_ok:
            raise FileNotFoundError(key)
        return None

    monkeypatch.setattr(Path, "mkdir", _mkdir, raising=False)
    monkeypatch.setattr(Path, "write_text", _write_text, raising=False)
    monkeypatch.setattr(Path, "read_text", _read_text, raising=False)
    monkeypatch.setattr(Path, "exists", _exists, raising=False)
    monkeypatch.setattr(Path, "replace", _replace, raising=False)
    monkeypatch.setattr(Path, "unlink", _unlink, raising=False)
    return storage


def test_history_store_roundtrip(monkeypatch):
    _install_fake_path_io(monkeypatch)
    history = [
        {
            "record_id": "rec-1",
            "generated_at": "2026-04-06 10:00:00",
            "mean_energy": -3.2,
            "protein_volume": 1234.5,
            "auto_detection_methods_used": "kvfinder,geometry-cluster",
            "auto_detection_status_summary": "kvfinder:used; geometry-cluster:used; consensus:consensus",
            "auto_detection_external_rows": 3,
            "auto_detection_external_sources": "M-CSA,UniProt",
            "top_pocket_id": "AutoPocket-1",
            "top_pocket_smart_rank_label": "高优先级",
            "top_pocket_smart_rank_score": 1.37,
            "top_pocket_reason": "多方法共识且覆盖热点残基",
            "top_joint_pocket_id": "AutoPocket-1",
            "top_joint_recommendation_label": "优先验证",
            "top_joint_recommendation_score": 0.812,
            "top_joint_reason": "口袋/界面/热点三重交集明显",
        }
    ]

    store_path = Path("history_test/analysis_history.json")
    session_state._save_history_to_disk(history, store_path)
    loaded = session_state._load_history_from_disk(store_path)

    assert loaded == history


def test_uploaded_inputs_cache_roundtrip(monkeypatch):
    _install_fake_path_io(monkeypatch)
    cache = {
        "pdb_files": [
            {"name": "a.pdb", "text": "ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND"},
            {"name": "b.pdb", "text": "ATOM      1  CA  GLY A   1      0.0 0.0 0.0  1.00 20.00           C\nEND"},
        ],
        "mmpbsa_files": [
            {"name": "energy.csv", "text": "chain,resid,delta_total\nA,1,-2.1"},
        ],
        "pocket_file": {"name": "pocket.csv", "text": "chain,resid,resname\nA,1,ALA"},
    }

    store_path = Path("history_test/uploaded_inputs.json")
    session_state._save_upload_cache_to_disk(cache, store_path)
    loaded = session_state._load_upload_cache_from_disk(store_path)

    assert loaded == cache


def test_history_store_falls_back_when_replace_is_denied(monkeypatch):
    storage = _install_fake_path_io(monkeypatch)
    history = [{"record_id": "rec-1", "generated_at": "2026-04-06 10:00:00"}]

    def _deny_replace(self, target):
        raise PermissionError("replace denied")

    monkeypatch.setattr(Path, "replace", _deny_replace, raising=False)

    store_path = Path("history_test/analysis_history.json")
    session_state._save_history_to_disk(history, store_path)
    loaded = session_state._load_history_from_disk(store_path)

    assert loaded == history
    assert json.loads(storage[str(store_path)]) == history


def test_clear_uploaded_inputs_cache_sections(monkeypatch):
    captured = {}
    cache = {
        "pdb_files": [{"name": "a.pdb", "text": "PDB"}],
        "mmpbsa_files": [{"name": "energy.csv", "text": "CSV"}],
        "pocket_file": {"name": "pocket.csv", "text": "POCKET"},
    }

    monkeypatch.setattr(session_state, "initialize_state", lambda: None)
    monkeypatch.setattr(session_state, "get_uploaded_inputs_cache", lambda: cache)

    def _fake_set_uploaded_inputs_cache(*, pdb_files, mmpbsa_files, pocket_file):
        captured["pdb_files"] = pdb_files
        captured["mmpbsa_files"] = mmpbsa_files
        captured["pocket_file"] = pocket_file

    monkeypatch.setattr(session_state, "set_uploaded_inputs_cache", _fake_set_uploaded_inputs_cache)

    session_state.clear_uploaded_inputs_cache_sections(clear_pdb=True, clear_mmpbsa=False, clear_pocket=True)

    assert captured["pdb_files"] == []
    assert captured["mmpbsa_files"] == cache["mmpbsa_files"]
    assert captured["pocket_file"] is None
