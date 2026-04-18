import json
from pathlib import Path

import streamlit as st
from datetime import datetime

from protein_visualizer.sample_data import MMPBSA_TEXT, PDB_TEXT


STATE_KEYS = {
    "pdb_text": "protein_visualizer_pdb_text",
    "mmpbsa_text": "protein_visualizer_mmpbsa_text",
    "energy_table": "protein_visualizer_energy_table",
    "annotation_table": "protein_visualizer_annotation_table",
    "pocket_table": "protein_visualizer_pocket_table",
    "pocket_summary": "protein_visualizer_pocket_summary",
    "joint_candidate_table": "protein_visualizer_joint_candidate_table",
    "atom_df": "protein_visualizer_atom_df",
    "energy_df": "protein_visualizer_energy_df",
    "color_mode": "protein_visualizer_color_mode",
    "history": "protein_visualizer_history",
    "uploaded_inputs": "protein_visualizer_uploaded_inputs",
}

ROOT_DIR = Path(__file__).resolve().parents[3]
HISTORY_STORE_PATH = ROOT_DIR / "data" / "analysis_history.json"
UPLOADED_INPUTS_STORE_PATH = ROOT_DIR / "data" / "uploaded_inputs.json"


def _empty_uploaded_inputs() -> dict:
    return {
        "pdb_files": [],
        "mmpbsa_files": [],
        "pocket_file": None,
    }


def _normalize_uploaded_file_entry(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    text = item.get("text")
    if text is None:
        return None
    name = str(item.get("name") or "uploaded_file")
    return {
        "name": name,
        "text": str(text),
    }


def _normalize_uploaded_inputs(payload: dict | None) -> dict:
    normalized = _empty_uploaded_inputs()
    if not isinstance(payload, dict):
        return normalized

    pdb_files = []
    for item in payload.get("pdb_files", []):
        entry = _normalize_uploaded_file_entry(item)
        if entry is not None:
            pdb_files.append(entry)

    mmpbsa_files = []
    for item in payload.get("mmpbsa_files", []):
        entry = _normalize_uploaded_file_entry(item)
        if entry is not None:
            mmpbsa_files.append(entry)

    pocket_entry = _normalize_uploaded_file_entry(payload.get("pocket_file"))

    normalized["pdb_files"] = pdb_files
    normalized["mmpbsa_files"] = mmpbsa_files
    normalized["pocket_file"] = pocket_entry
    return normalized


def _load_upload_cache_from_disk(path: Path | None = None) -> dict:
    store_path = path or UPLOADED_INPUTS_STORE_PATH
    try:
        if not store_path.exists():
            return _empty_uploaded_inputs()
        payload = json.loads(store_path.read_text(encoding="utf-8"))
        return _normalize_uploaded_inputs(payload)
    except Exception:
        pass
    return _empty_uploaded_inputs()


def _write_json_atomically(payload: object, store_path: Path) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = store_path.with_suffix(store_path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_path.write_text(text, encoding="utf-8")
    try:
        temp_path.replace(store_path)
    except PermissionError:
        store_path.write_text(text, encoding="utf-8")
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def _save_upload_cache_to_disk(cache: dict, path: Path | None = None) -> None:
    store_path = path or UPLOADED_INPUTS_STORE_PATH
    normalized = _normalize_uploaded_inputs(cache)
    _write_json_atomically(normalized, store_path)


def _load_history_from_disk(path: Path | None = None) -> list[dict]:
    store_path = path or HISTORY_STORE_PATH
    try:
        if not store_path.exists():
            return []
        payload = json.loads(store_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    except Exception:
        pass
    return []


def _save_history_to_disk(history: list[dict], path: Path | None = None) -> None:
    store_path = path or HISTORY_STORE_PATH
    _write_json_atomically(history, store_path)


def initialize_state() -> None:
    if STATE_KEYS["pdb_text"] not in st.session_state:
        st.session_state[STATE_KEYS["pdb_text"]] = PDB_TEXT
    if STATE_KEYS["mmpbsa_text"] not in st.session_state:
        st.session_state[STATE_KEYS["mmpbsa_text"]] = MMPBSA_TEXT
    if STATE_KEYS["history"] not in st.session_state:
        st.session_state[STATE_KEYS["history"]] = _load_history_from_disk()
    if STATE_KEYS["uploaded_inputs"] not in st.session_state:
        st.session_state[STATE_KEYS["uploaded_inputs"]] = _load_upload_cache_from_disk()


def set_analysis_state(
    pdb_text,
    mmpbsa_text,
    atom_df,
    energy_df,
    energy_table,
    annotation_table=None,
    pocket_table=None,
    pocket_summary=None,
    joint_candidate_table=None,
    color_mode=None,
) -> None:
    st.session_state[STATE_KEYS["pdb_text"]] = pdb_text
    st.session_state[STATE_KEYS["mmpbsa_text"]] = mmpbsa_text
    st.session_state[STATE_KEYS["atom_df"]] = atom_df
    st.session_state[STATE_KEYS["energy_df"]] = energy_df
    st.session_state[STATE_KEYS["energy_table"]] = energy_table
    if annotation_table is not None:
        st.session_state[STATE_KEYS["annotation_table"]] = annotation_table
    if pocket_table is not None:
        st.session_state[STATE_KEYS["pocket_table"]] = pocket_table
    if pocket_summary is not None:
        st.session_state[STATE_KEYS["pocket_summary"]] = pocket_summary
    if joint_candidate_table is not None:
        st.session_state[STATE_KEYS["joint_candidate_table"]] = joint_candidate_table
    if color_mode is not None:
        st.session_state[STATE_KEYS["color_mode"]] = color_mode


def get_current_pdb_text() -> str:
    initialize_state()
    return st.session_state[STATE_KEYS["pdb_text"]]


def get_current_mmpbsa_text() -> str:
    initialize_state()
    return st.session_state[STATE_KEYS["mmpbsa_text"]]


def get_current_energy_table():
    return st.session_state.get(STATE_KEYS["energy_table"])


def get_current_annotation_table():
    return st.session_state.get(STATE_KEYS["annotation_table"])


def get_current_pocket_table():
    return st.session_state.get(STATE_KEYS["pocket_table"])


def get_current_pocket_summary():
    return st.session_state.get(STATE_KEYS["pocket_summary"])


def get_current_joint_candidate_table():
    return st.session_state.get(STATE_KEYS["joint_candidate_table"])


def get_current_color_mode() -> str:
    return st.session_state.get(STATE_KEYS["color_mode"], "按DELTA TOTAL 热度")


def get_current_atom_df():
    return st.session_state.get(STATE_KEYS["atom_df"])


def get_current_energy_df():
    return st.session_state.get(STATE_KEYS["energy_df"])


def append_history_record(record: dict, max_items: int = 10) -> None:
    initialize_state()
    history = st.session_state[STATE_KEYS["history"]]
    enriched_record = {
        "record_id": record.get("record_id", datetime.now().strftime("%Y%m%d%H%M%S%f")),
        **record,
    }
    if history and history[0].get("generated_at") == enriched_record.get("generated_at"):
        return
    history.insert(0, enriched_record)
    trimmed_history = history[:max_items]
    st.session_state[STATE_KEYS["history"]] = trimmed_history
    try:
        _save_history_to_disk(trimmed_history)
    except Exception:
        pass


def get_history_records():
    initialize_state()
    return st.session_state[STATE_KEYS["history"]]


def get_history_store_path() -> Path:
    return HISTORY_STORE_PATH


def reload_history_from_disk() -> list[dict]:
    history = _load_history_from_disk()
    st.session_state[STATE_KEYS["history"]] = history
    return history


def get_uploaded_inputs_cache() -> dict:
    initialize_state()
    return _normalize_uploaded_inputs(st.session_state.get(STATE_KEYS["uploaded_inputs"]))


def set_uploaded_inputs_cache(
    *,
    pdb_files: list[dict] | None = None,
    mmpbsa_files: list[dict] | None = None,
    pocket_file: dict | None = None,
) -> None:
    initialize_state()
    cache = _normalize_uploaded_inputs(
        {
            "pdb_files": pdb_files or [],
            "mmpbsa_files": mmpbsa_files or [],
            "pocket_file": pocket_file,
        }
    )
    st.session_state[STATE_KEYS["uploaded_inputs"]] = cache
    try:
        _save_upload_cache_to_disk(cache)
    except Exception:
        pass


def clear_uploaded_inputs_cache() -> None:
    set_uploaded_inputs_cache(pdb_files=[], mmpbsa_files=[], pocket_file=None)


def clear_uploaded_inputs_cache_sections(
    *,
    clear_pdb: bool = False,
    clear_mmpbsa: bool = False,
    clear_pocket: bool = False,
) -> None:
    initialize_state()
    cache = get_uploaded_inputs_cache()
    set_uploaded_inputs_cache(
        pdb_files=[] if clear_pdb else cache.get("pdb_files", []),
        mmpbsa_files=[] if clear_mmpbsa else cache.get("mmpbsa_files", []),
        pocket_file=None if clear_pocket else cache.get("pocket_file"),
    )
