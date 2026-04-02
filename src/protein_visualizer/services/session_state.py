import streamlit as st
from datetime import datetime

from protein_visualizer.sample_data import MMPBSA_TEXT, PDB_TEXT


STATE_KEYS = {
    "pdb_text": "protein_visualizer_pdb_text",
    "mmpbsa_text": "protein_visualizer_mmpbsa_text",
    "energy_table": "protein_visualizer_energy_table",
    "atom_df": "protein_visualizer_atom_df",
    "energy_df": "protein_visualizer_energy_df",
    "history": "protein_visualizer_history",
}


def initialize_state() -> None:
    if STATE_KEYS["pdb_text"] not in st.session_state:
        st.session_state[STATE_KEYS["pdb_text"]] = PDB_TEXT
    if STATE_KEYS["mmpbsa_text"] not in st.session_state:
        st.session_state[STATE_KEYS["mmpbsa_text"]] = MMPBSA_TEXT
    if STATE_KEYS["history"] not in st.session_state:
        st.session_state[STATE_KEYS["history"]] = []


def set_analysis_state(pdb_text, mmpbsa_text, atom_df, energy_df, energy_table) -> None:
    st.session_state[STATE_KEYS["pdb_text"]] = pdb_text
    st.session_state[STATE_KEYS["mmpbsa_text"]] = mmpbsa_text
    st.session_state[STATE_KEYS["atom_df"]] = atom_df
    st.session_state[STATE_KEYS["energy_df"]] = energy_df
    st.session_state[STATE_KEYS["energy_table"]] = energy_table


def get_current_pdb_text() -> str:
    initialize_state()
    return st.session_state[STATE_KEYS["pdb_text"]]


def get_current_mmpbsa_text() -> str:
    initialize_state()
    return st.session_state[STATE_KEYS["mmpbsa_text"]]


def get_current_energy_table():
    return st.session_state.get(STATE_KEYS["energy_table"])


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
    st.session_state[STATE_KEYS["history"]] = history[:max_items]


def get_history_records():
    initialize_state()
    return st.session_state[STATE_KEYS["history"]]
