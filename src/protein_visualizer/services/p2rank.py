from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd


P2RANK_POCKET_COLUMNS = [
    "pocket_label",
    "pocket_rank",
    "pocket_score",
    "pocket_probability",
    "center_x",
    "center_y",
    "center_z",
    "residue_list",
]

P2RANK_RESIDUE_COLUMNS = [
    "pocket_label",
    "pocket_rank",
    "chain",
    "resid",
    "resname",
    "residue_score",
    "residue_probability",
]


def _empty_p2rank_pocket_df() -> pd.DataFrame:
    return pd.DataFrame(columns=P2RANK_POCKET_COLUMNS)


def _empty_p2rank_residue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=P2RANK_RESIDUE_COLUMNS)


def _normalized_column_map(frame: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", "", str(column).strip().lower())
        if normalized and normalized not in mapping:
            mapping[normalized] = str(column)
    return mapping


def _find_column(frame: pd.DataFrame, *aliases: str) -> Optional[str]:
    mapping = _normalized_column_map(frame)
    for alias in aliases:
        normalized = re.sub(r"[^a-z0-9]+", "", str(alias).strip().lower())
        if normalized in mapping:
            return mapping[normalized]
    return None


def _to_int(value: object) -> Optional[int]:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return float(default)
        return float(text)
    except Exception:
        return float(default)


def parse_p2rank_predictions_csv(text: str) -> pd.DataFrame:
    if not str(text or "").strip():
        return _empty_p2rank_pocket_df()

    frame = pd.read_csv(pd.io.common.StringIO(str(text)))
    if frame.empty:
        return _empty_p2rank_pocket_df()

    rank_col = _find_column(frame, "rank", "pocket_rank", "id")
    score_col = _find_column(frame, "score", "pocket_score")
    probability_col = _find_column(frame, "probability", "pocket_probability", "prob")
    name_col = _find_column(frame, "name", "pocket", "pocket_id", "pocket_label")
    residue_list_col = _find_column(frame, "residues", "residue_ids", "residue_list", "residue_labels", "adjacent_residues")
    center_x_col = _find_column(frame, "center_x", "centerx", "x")
    center_y_col = _find_column(frame, "center_y", "centery", "y")
    center_z_col = _find_column(frame, "center_z", "centerz", "z")

    records = []
    for row_index, row in enumerate(frame.to_dict("records"), start=1):
        pocket_rank = _to_int(row.get(rank_col) if rank_col else None)
        if pocket_rank is None:
            pocket_rank = row_index

        raw_label = str(row.get(name_col, "") if name_col else "").strip()
        pocket_label = raw_label or f"P2Rank-{pocket_rank}"

        records.append(
            {
                "pocket_label": pocket_label,
                "pocket_rank": int(pocket_rank),
                "pocket_score": _to_float(row.get(score_col) if score_col else None, default=0.0),
                "pocket_probability": _to_float(row.get(probability_col) if probability_col else None, default=0.0),
                "center_x": _to_float(row.get(center_x_col) if center_x_col else None, default=0.0),
                "center_y": _to_float(row.get(center_y_col) if center_y_col else None, default=0.0),
                "center_z": _to_float(row.get(center_z_col) if center_z_col else None, default=0.0),
                "residue_list": str(row.get(residue_list_col, "") if residue_list_col else "").strip(),
            }
        )

    if not records:
        return _empty_p2rank_pocket_df()
    return pd.DataFrame(records, columns=P2RANK_POCKET_COLUMNS)


def _parse_residue_list_text(text: str) -> list[Tuple[str, int, str]]:
    residue_text = str(text or "").strip()
    if not residue_text:
        return []

    matches: list[Tuple[str, int, str]] = []
    patterns = [
        re.compile(r"\b([A-Za-z]{3})\s+([A-Za-z0-9])[:_\s]+(-?\d+)\b"),
        re.compile(r"\b([A-Za-z0-9])[:_\s]+(-?\d+)\s+([A-Za-z]{3})\b"),
        re.compile(r"\b([A-Za-z0-9])[:_]?(-?\d+)\b"),
    ]
    for pattern_index, pattern in enumerate(patterns):
        for match in pattern.finditer(residue_text):
            groups = match.groups()
            if pattern_index == 0:
                resname, chain, resid = groups
            elif pattern_index == 1:
                chain, resid, resname = groups
            else:
                chain, resid = groups
                resname = ""

            parsed_resid = _to_int(resid)
            if parsed_resid is None:
                continue
            matches.append((str(chain).strip() or "A", int(parsed_resid), str(resname).strip().upper()))

        if matches:
            break

    deduped: list[Tuple[str, int, str]] = []
    seen: set[Tuple[str, int]] = set()
    for chain, resid, resname in matches:
        key = (chain, resid)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((chain, resid, resname))
    return deduped


def _prediction_rows_to_residue_df(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df is None or predictions_df.empty:
        return _empty_p2rank_residue_df()

    rows = []
    for row in predictions_df.itertuples(index=False):
        residues = _parse_residue_list_text(getattr(row, "residue_list", ""))
        if not residues:
            continue
        for chain, resid, resname in residues:
            rows.append(
                {
                    "pocket_label": str(getattr(row, "pocket_label", "")).strip(),
                    "pocket_rank": int(getattr(row, "pocket_rank", 0) or 0),
                    "chain": chain,
                    "resid": int(resid),
                    "resname": resname,
                    "residue_score": float(getattr(row, "pocket_score", 0.0) or 0.0),
                    "residue_probability": float(getattr(row, "pocket_probability", 0.0) or 0.0),
                }
            )

    if not rows:
        return _empty_p2rank_residue_df()
    return pd.DataFrame(rows, columns=P2RANK_RESIDUE_COLUMNS)


def parse_p2rank_residues_csv(text: str) -> pd.DataFrame:
    if not str(text or "").strip():
        return _empty_p2rank_residue_df()

    frame = pd.read_csv(pd.io.common.StringIO(str(text)))
    if frame.empty:
        return _empty_p2rank_residue_df()

    rank_col = _find_column(frame, "pocket_rank", "rank", "cluster")
    pocket_col = _find_column(frame, "pocket", "pocket_label", "pocket_id", "name")
    chain_col = _find_column(frame, "chain", "chain_id", "auth_asym_id")
    resid_col = _find_column(frame, "resid", "residue", "residue_number", "residue_id", "seq_id", "author_residue_number")
    resname_col = _find_column(frame, "resname", "residue_name", "aa", "residue_type")
    score_col = _find_column(frame, "score", "residue_score", "prediction_score")
    probability_col = _find_column(frame, "probability", "residue_probability", "prob")
    label_col = _find_column(frame, "residue_label", "label")

    records = []
    for row_index, row in enumerate(frame.to_dict("records"), start=1):
        raw_chain = str(row.get(chain_col, "") if chain_col else "").strip()
        raw_resid = _to_int(row.get(resid_col) if resid_col else None)
        raw_resname = str(row.get(resname_col, "") if resname_col else "").strip().upper()

        if raw_resid is None and label_col:
            label_matches = _parse_residue_list_text(str(row.get(label_col, "") or ""))
            if label_matches:
                raw_chain, raw_resid, parsed_resname = label_matches[0]
                if not raw_resname:
                    raw_resname = parsed_resname

        if raw_resid is None:
            continue

        pocket_rank = _to_int(row.get(rank_col) if rank_col else None)
        if pocket_rank is None:
            pocket_rank = row_index

        raw_pocket = str(row.get(pocket_col, "") if pocket_col else "").strip()
        pocket_label = raw_pocket or f"P2Rank-{pocket_rank}"

        records.append(
            {
                "pocket_label": pocket_label,
                "pocket_rank": int(pocket_rank),
                "chain": raw_chain or "A",
                "resid": int(raw_resid),
                "resname": raw_resname,
                "residue_score": _to_float(row.get(score_col) if score_col else None, default=0.0),
                "residue_probability": _to_float(row.get(probability_col) if probability_col else None, default=0.0),
            }
        )

    if not records:
        return _empty_p2rank_residue_df()
    return pd.DataFrame(records, columns=P2RANK_RESIDUE_COLUMNS)


def _resolve_p2rank_runner(executable: Optional[str] = None) -> Optional[Path]:
    candidates: list[Path] = []
    if executable:
        candidates.append(Path(str(executable)).expanduser())

    env_script = os.getenv("P2RANK_SCRIPT")
    env_home = os.getenv("P2RANK_HOME")
    if env_script:
        candidates.append(Path(env_script).expanduser())
    if env_home:
        home_path = Path(env_home).expanduser()
        candidates.extend(
            [
                home_path / "prank",
                home_path / "prank.sh",
                home_path / "prank.bat",
                home_path / "p2rank.jar",
            ]
        )

    for candidate in candidates:
        if candidate.is_dir():
            for name in ("prank", "prank.sh", "prank.bat", "p2rank.jar"):
                nested = candidate / name
                if nested.exists():
                    return nested
        elif candidate.exists():
            return candidate
    return None


def _build_command(
    runner: Path,
    *,
    pdb_path: Path,
    output_dir: Path,
    profile: str,
) -> list[str]:
    suffix = runner.suffix.lower()
    base_args = ["predict", "-f", str(pdb_path), "-o", str(output_dir), "-visualizations", "0"]
    cleaned_profile = str(profile or "").strip().lower()
    if cleaned_profile and cleaned_profile not in {"default", "auto"}:
        base_args.extend(["-c", cleaned_profile])

    if suffix == ".jar":
        return ["java", "-jar", str(runner), *base_args]
    if suffix in {".bat", ".cmd"}:
        return ["cmd.exe", "/c", str(runner), *base_args]
    return [str(runner), *base_args]


def _select_csv(paths: Sequence[Path], *keywords: str) -> Optional[Path]:
    lowered_keywords = [keyword.lower() for keyword in keywords]
    for path in paths:
        lowered_name = path.name.lower()
        if all(keyword in lowered_name for keyword in lowered_keywords):
            return path
    return None


def run_p2rank(
    pdb_text: str,
    *,
    executable: Optional[str] = None,
    profile: str = "default",
    timeout_sec: float = 180.0,
    output_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not str(pdb_text or "").strip():
        return _empty_p2rank_pocket_df(), _empty_p2rank_residue_df(), {"status": "empty-input"}

    runner = _resolve_p2rank_runner(executable)
    if runner is None:
        return _empty_p2rank_pocket_df(), _empty_p2rank_residue_df(), {"status": "unavailable"}

    temp_dir_obj = None
    working_dir = Path(output_dir).expanduser() if output_dir else None
    if working_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="p2rank_")
        working_dir = Path(temp_dir_obj.name)
    working_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = working_dir / "input_structure.pdb"
    output_root = working_dir / "p2rank_output"
    output_root.mkdir(parents=True, exist_ok=True)
    pdb_path.write_text(str(pdb_text), encoding="utf-8")

    command = _build_command(
        runner,
        pdb_path=pdb_path,
        output_dir=output_root,
        profile=profile,
    )

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_sec)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()
        return _empty_p2rank_pocket_df(), _empty_p2rank_residue_df(), {
            "status": "failed",
            "reason": str(exc),
        }

    csv_files = sorted(output_root.rglob("*.csv"))
    prediction_csv = _select_csv(csv_files, "prediction")
    residue_csv = _select_csv(csv_files, "residue")

    pocket_df = _empty_p2rank_pocket_df()
    residue_df = _empty_p2rank_residue_df()
    if prediction_csv and prediction_csv.exists():
        pocket_df = parse_p2rank_predictions_csv(prediction_csv.read_text(encoding="utf-8", errors="ignore"))
    if residue_csv and residue_csv.exists():
        residue_df = parse_p2rank_residues_csv(residue_csv.read_text(encoding="utf-8", errors="ignore"))
    if residue_df.empty and not pocket_df.empty:
        residue_df = _prediction_rows_to_residue_df(pocket_df)

    metadata = {
        "status": "ok" if completed.returncode == 0 else "failed",
        "return_code": int(completed.returncode),
        "stdout": str(completed.stdout or "").strip(),
        "stderr": str(completed.stderr or "").strip(),
        "prediction_rows": str(len(pocket_df)),
        "residue_rows": str(len(residue_df)),
    }

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()
    return pocket_df, residue_df, metadata
