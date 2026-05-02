#!/usr/bin/env python3.9
"""
world_ema_viz.py â€” schema-flexible WORLD_* log review (lazy-rendered).

Views: Group / Individual / 3D.  Only the visible plots are mounted; hidden
ones are Plotly.purge()'d so WebGL contexts are released.
"""

import argparse
import codecs
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import hashlib
import html
import io
import json
import math
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

AXES = ("X", "Y", "Z")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
SKELETON_RAW_COLOR = "#374151"
SKELETON_LOGGED_FILTERED_COLOR = "#2563eb"
SKELETON_SIM_FILTERED_COLOR = "#f97316"


def _emit_progress(progress_callback: Optional[Callable[[str], None]], message: str) -> None:
    if progress_callback:
        progress_callback(message)

SKELETON_CONNECTIONS = (
    ("nose", "leftEyeInner"),
    ("leftEyeInner", "leftEye"),
    ("leftEye", "leftEyeOuter"),
    ("leftEyeOuter", "leftEar"),
    ("nose", "rightEyeInner"),
    ("rightEyeInner", "rightEye"),
    ("rightEye", "rightEyeOuter"),
    ("rightEyeOuter", "rightEar"),
    ("leftMouth", "rightMouth"),
    ("leftShoulder", "rightShoulder"),
    ("leftShoulder", "leftElbow"),
    ("leftElbow", "leftWrist"),
    ("leftWrist", "leftPinky"),
    ("leftWrist", "leftIndex"),
    ("leftWrist", "leftThumb"),
    ("leftPinky", "leftIndex"),
    ("rightShoulder", "rightElbow"),
    ("rightElbow", "rightWrist"),
    ("rightWrist", "rightPinky"),
    ("rightWrist", "rightIndex"),
    ("rightWrist", "rightThumb"),
    ("rightPinky", "rightIndex"),
    ("leftShoulder", "leftHip"),
    ("rightShoulder", "rightHip"),
    ("leftHip", "rightHip"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("leftAnkle", "leftHeel"),
    ("leftHeel", "leftFootIndex"),
    ("leftAnkle", "leftFootIndex"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
    ("rightAnkle", "rightHeel"),
    ("rightHeel", "rightFootIndex"),
    ("rightAnkle", "rightFootIndex"),
)


@dataclass
class PoseDetectionSettings:
    Q: float = 1.0
    R: float = 1.0
    inFrameLikelihoodThreshold: float = 0.5
    numPreviousLocations: int = 5
    EMA: float = 0.1
    adaptiveQScalingFactor: float = 1.0
    dynamicRInnovationSlope: float = 0.0
    dynamicRInnovationThreshold: float = 1.0
    velocityAlphaMin: float = 0.1
    velocityAlphaMax: float = 0.9
    velocityAlphaMidpoint: float = 0.5
    velocityAlphaSlope: float = 1.0
    positionAlphaMin: float = 0.1
    positionAlphaMax: float = 0.9
    positionAlphaDecay: float = 0.95
    accelerationAlphaMin: float = 0.1
    accelerationAlphaMax: float = 0.9
    accelerationAlphaMidpoint: float = 0.5
    accelerationAlphaSlope: float = 1.0
    accelerationAlphaDecay: float = 0.95
    baseProcessNoiseAccel: float = 0.01
    accelerationDecayRate: float = 0.9
    smoothingDecayTimeConstant: float = 0.1
    updateDecayTimeConstant: float = 0.1
    accelerationBoundsDecayFactor: float = 0.9
    accelerationInnovationContributionFactor: float = 0.5
    accelerationBoundsThreshold: float = 1.0
    accelerationSaturationThreshold: float = 10.0
    accelerationSaturationNoiseReduction: float = 0.5
    accelerationAdaptiveScalingFactor: float = 1.0
    gainAttenuationRate: float = 0.0
    positionLockThreshold: float = 0.0
    confidenceRScaling: float = 0.0
    legacyConfidenceThreshold: float = 0.5
    visibilityThreshold: float = 0.5
    presenceThreshold: float = 0.5
    highPresenceThreshold: float = 0.75
    lowVisibilityThreshold: float = 0.2
    worldSpaceEMA: float = 0.37


@dataclass
class FilterResult:
    position: float
    velocity: float
    acceleration: float
    covPosPos: float
    covPosVel: float
    covPosAccel: float
    covVelPos: float
    covVelVel: float
    covVelAccel: float
    covAccelAccel: float
    innovationNormSquared: float


@dataclass
class FilterAxisState:
    position: float
    velocity: float = 0.0
    acceleration: float = 0.0
    covPosPos: float = 1.0
    covPosVel: float = 0.0
    covPosAccel: float = 0.0
    covVelPos: float = 0.0
    covVelVel: float = 1.0
    covVelAccel: float = 0.0
    covAccelPos: float = 0.0
    covAccelVel: float = 0.0
    covAccelAccel: float = 1.0
    innovationNormSquared: float = 0.0

    @classmethod
    def initialized(cls, measured_pos: float) -> "FilterAxisState":
        return cls(position=measured_pos)

    @classmethod
    def from_result(cls, result: FilterResult) -> "FilterAxisState":
        return cls(
            position=result.position,
            velocity=result.velocity,
            acceleration=result.acceleration,
            covPosPos=result.covPosPos,
            covPosVel=result.covPosVel,
            covPosAccel=result.covPosAccel,
            covVelPos=result.covVelPos,
            covVelVel=result.covVelVel,
            covVelAccel=result.covVelAccel,
            covAccelPos=result.covPosAccel,
            covAccelVel=result.covVelAccel,
            covAccelAccel=result.covAccelAccel,
            innovationNormSquared=result.innovationNormSquared,
        )


class WorldSpaceKalmanFilter:
    """Runtime adapter for the ignored proprietary Kalman implementation."""

    def __init__(self, settings: PoseDetectionSettings):
        self.settings = settings
        try:
            from world_space_kalman_filter_private import (
                WorldSpaceKalmanFilter as PrivateWorldSpaceKalmanFilter,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Missing world_space_kalman_filter_private.py. "
                "This proprietary file is intentionally git-ignored; restore it "
                "locally to run simulation playback."
            ) from exc
        self._impl = PrivateWorldSpaceKalmanFilter(settings)

    def filter_axis(self, *args, **kwargs) -> FilterResult:
        return self._impl.filter_axis(*args, **kwargs)


def parse_pose_detection_settings(values: Mapping[str, Any]) -> PoseDetectionSettings:
    parsed = asdict(PoseDetectionSettings())
    valid_names = {f.name for f in fields(PoseDetectionSettings)}
    for name, value in values.items():
        if name not in valid_names:
            continue
        text = str(value).strip()
        if name == "numPreviousLocations":
            number = float(text)
            if not math.isfinite(number) or not number.is_integer():
                raise ValueError(f"{name} must be a finite integer")
            parsed[name] = int(number)
        else:
            number = float(text)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            parsed[name] = number
    return PoseDetectionSettings(**parsed)


def _settings_field_names() -> List[str]:
    return [f.name for f in fields(PoseDetectionSettings)]


def _clean_numeric_literal(value: Any) -> str:
    text = str(value).strip().rstrip(",")
    text = re.sub(r"[fFdD]$", "", text)
    return text.replace("_", "")


def _coerce_settings_values(values: Mapping[str, Any]) -> Dict[str, Any]:
    valid_names = set(_settings_field_names())
    return {
        name: _clean_numeric_literal(value)
        for name, value in values.items()
        if name in valid_names and str(value).strip()
    }


def _looks_like_settings_map(value: Any) -> bool:
    return (
        isinstance(value, Mapping) and
        any(name in value for name in _settings_field_names())
    )


def _json_settings_presets(value: Any, label: str = "Imported") -> Dict[str, Dict[str, Any]]:
    presets: Dict[str, Dict[str, Any]] = {}
    if _looks_like_settings_map(value):
        presets[label] = _coerce_settings_values(value)
    if isinstance(value, Mapping):
        if _looks_like_settings_map(value.get("poseDetection")):
            presets[label] = _coerce_settings_values(value["poseDetection"])
        for key, child in value.items():
            child_label = str(key)
            if _looks_like_settings_map(child):
                presets[child_label] = _coerce_settings_values(child)
            elif isinstance(child, Mapping) and _looks_like_settings_map(
                child.get("poseDetection")
            ):
                presets[child_label] = _coerce_settings_values(child["poseDetection"])
            elif isinstance(child, (Mapping, list)):
                presets.update(_json_settings_presets(child, child_label))
    elif isinstance(value, list):
        for index, child in enumerate(value, start=1):
            child_label = label
            if isinstance(child, Mapping):
                child_label = str(
                    child.get("name") or child.get("preset") or f"{label} {index}"
                )
            presets.update(_json_settings_presets(child, child_label))
    return presets


def _csv_settings_presets(text: str) -> Dict[str, Dict[str, Any]]:
    presets: Dict[str, Dict[str, Any]] = {}
    try:
        sample = text[:2048]
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    try:
        rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    except csv.Error:
        return {}
    if not rows:
        return {}

    lower_to_field = {name.lower(): name for name in _settings_field_names()}
    fieldnames = [name or "" for name in (rows[0].keys() if rows else [])]
    lower_headers = {name.lower(): name for name in fieldnames}
    name_column = next(
        (lower_headers[key] for key in ("name", "field", "key", "coefficient")
         if key in lower_headers),
        None,
    )
    value_column = next(
        (lower_headers[key] for key in ("value", "setting", "coefficientvalue")
         if key in lower_headers),
        None,
    )
    preset_column = next(
        (lower_headers[key] for key in ("preset", "configuration", "config")
         if key in lower_headers),
        None,
    )

    if name_column and value_column:
        for row in rows:
            field_name = lower_to_field.get(str(row.get(name_column, "")).strip().lower())
            if not field_name:
                continue
            preset_name = str(row.get(preset_column, "")).strip() if preset_column else ""
            preset_name = preset_name or "CSV"
            presets.setdefault(preset_name, {})[field_name] = row.get(value_column, "")
        return {name: _coerce_settings_values(values)
                for name, values in presets.items()}

    coefficient_headers = {
        header: lower_to_field[header.lower()]
        for header in fieldnames
        if header.lower() in lower_to_field
    }
    if coefficient_headers:
        for index, row in enumerate(rows, start=1):
            preset_name = (
                str(row.get(preset_column, "")).strip() if preset_column else ""
            ) or str(row.get("name", "")).strip() or f"CSV row {index}"
            presets[preset_name] = _coerce_settings_values(
                {
                    field_name: row.get(header, "")
                    for header, field_name in coefficient_headers.items()
                }
            )
    return presets


def _balanced_parenthesized_content(text: str, open_paren_index: int) -> str:
    depth = 0
    for index in range(open_paren_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_index + 1:index]
    return text[open_paren_index + 1:]


def _extract_field_assignments(text: str) -> Dict[str, str]:
    field_pattern = "|".join(re.escape(name) for name in _settings_field_names())
    number_pattern = r"[-+]?(?:\d[\d_]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    assignments = {}
    for match in re.finditer(
        rf"\b({field_pattern})\s*=\s*({number_pattern})\s*[fFdD]?",
        text,
    ):
        assignments[match.group(1)] = _clean_numeric_literal(match.group(2))
    return assignments


def _kotlin_settings_presets(text: str) -> Dict[str, Dict[str, Any]]:
    presets: Dict[str, Dict[str, Any]] = {}
    config_pattern = re.compile(
        r"(?:@JvmField\s+)?val\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*FilterSettings\s*\(",
        re.MULTILINE,
    )
    for match in config_pattern.finditer(text):
        name = match.group(1)
        pose_match = re.search(r"PoseDetectionSettings\s*\(", text[match.end():])
        if not pose_match:
            continue
        open_paren_index = match.end() + pose_match.end() - 1
        content = _balanced_parenthesized_content(text, open_paren_index)
        values = _extract_field_assignments(content)
        if values:
            presets[name] = values
    if not presets:
        values = _extract_field_assignments(text)
        if values:
            presets["Imported"] = values
    return presets


def extract_pose_detection_settings_presets(
    text: str,
    source_name: str = "Imported",
) -> Dict[str, Dict[str, Any]]:
    """Extract PoseDetectionSettings presets from JSON, CSV, Kotlin, or text."""
    text = text.strip()
    if not text:
        return {}

    presets: Dict[str, Dict[str, Any]] = {}
    try:
        presets.update(_json_settings_presets(json.loads(text), source_name))
    except json.JSONDecodeError:
        pass

    presets.update(_csv_settings_presets(text))
    presets.update(_kotlin_settings_presets(text))
    return {
        name: values for name, values in presets.items()
        if _coerce_settings_values(values)
    }


def parse_pose_detection_settings_document(path: Path) -> Dict[str, PoseDetectionSettings]:
    text = Path(path).read_text(encoding="utf-8-sig")
    presets = extract_pose_detection_settings_presets(text, Path(path).stem)
    return {
        name: parse_pose_detection_settings(values)
        for name, values in presets.items()
    }


def compute_dt(previous_frame_no: Optional[int], current_frame_no: int, fps: float) -> float:
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps must be a positive finite value")
    frame_duration = 1.0 / fps
    if previous_frame_no is None:
        return frame_duration
    return max((current_frame_no - previous_frame_no) / fps, frame_duration)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _decode_bytes(raw: bytes) -> str:
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return raw.decode("utf-16")
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig")
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_log(path: Path, tag_re: str = r"WORLD_\w+", extra_names=None) -> pd.DataFrame:
    """
    Contract:  <TAG>|<frameNo>|<landmark>|<g0>|<g1>|...|<gN>
      g0 = raw x,y,z (required)   g1 = filt x,y,z (required)
      gN(last) = alpha if scalar (else 0.0)
      g2..gN-1 = extras (CSV-only)
    """
    payload_re = re.compile(
        rf"(?P<tag>{tag_re})\|(?P<frame>\d+)\|(?P<lm>\w+)\|(?P<rest>\S.*)"
    )
    extra_names = extra_names or []
    text = _decode_bytes(path.read_bytes())

    records = []
    for line in text.splitlines():
        m = payload_re.search(line)
        if not m:
            continue
        tag, frame, lm = m["tag"], int(m["frame"]), m["lm"]
        groups = []
        for chunk in m["rest"].strip().split("|"):
            nums = NUM_RE.findall(chunk)
            if nums:
                groups.append([float(x) for x in nums])
        if len(groups) < 2 or len(groups[0]) != 3 or len(groups[1]) != 3:
            continue
        rec = {"tag": tag, "frameNo": frame, "landmark": lm,
               "rawX": groups[0][0], "rawY": groups[0][1], "rawZ": groups[0][2],
               "filtX": groups[1][0], "filtY": groups[1][1], "filtZ": groups[1][2]}
        tail = len(groups)
        if tail >= 3 and len(groups[-1]) == 1:
            rec["alpha"] = groups[-1][0]; tail -= 1
        else:
            rec["alpha"] = 0.0
        for j, gi in enumerate(range(2, tail)):
            name = extra_names[j] if j < len(extra_names) else f"extra{j}"
            vals = groups[gi]
            if len(vals) == 3:
                rec[f"{name}X"], rec[f"{name}Y"], rec[f"{name}Z"] = vals
            elif len(vals) == 1:
                rec[name] = vals[0]
            else:
                for k, v in enumerate(vals):
                    rec[f"{name}_{k}"] = v
        records.append(rec)

    if not records:
        raise ValueError(f"No lines matching tag /{tag_re}/ in {path}")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["frameNo", "landmark", "tag", "alpha"], keep="last")
    df["variant"] = [f"{t} Î±={a:g}" for t, a in zip(df["tag"], df["alpha"])]
    df = df.sort_values(["landmark", "variant", "frameNo"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Skeleton playback and simulation helpers
# ---------------------------------------------------------------------------
def sorted_frame_numbers(df: pd.DataFrame) -> List[int]:
    return [int(v) for v in sorted(df["frameNo"].unique())]


def group_frames(df: pd.DataFrame) -> List[Dict[str, Any]]:
    frames = []
    for frame_no in sorted_frame_numbers(df):
        g = df[df["frameNo"] == frame_no].sort_values("landmark")
        landmarks = {}
        for _, row in g.iterrows():
            landmarks[str(row["landmark"])] = row.to_dict()
        frames.append({"frameNo": frame_no, "landmarks": landmarks})
    return frames


def _coord(row: Mapping[str, Any], prefix: str, axis: str) -> float:
    return float(row[f"{prefix}{axis}"])


def build_skeleton_segments(
    frame_landmarks: Mapping[str, Mapping[str, Any]],
    prefix: str = "raw",
    connections: Tuple[Tuple[str, str], ...] = SKELETON_CONNECTIONS,
) -> List[Dict[str, Any]]:
    segments = []
    for start, end in connections:
        if start not in frame_landmarks or end not in frame_landmarks:
            continue
        start_row = frame_landmarks[start]
        end_row = frame_landmarks[end]
        keys = [f"{prefix}{axis}" for axis in AXES]
        if not all(key in start_row and key in end_row for key in keys):
            continue
        segments.append({
            "start": start,
            "end": end,
            "x": [_coord(start_row, prefix, "X"), _coord(end_row, prefix, "X")],
            "y": [_coord(start_row, prefix, "Y"), _coord(end_row, prefix, "Y")],
            "z": [_coord(start_row, prefix, "Z"), _coord(end_row, prefix, "Z")],
        })
    return segments


def build_logged_skeleton_trace_structures(
    frame: Mapping[str, Any],
    visibility: str = "both",
) -> List[Dict[str, Any]]:
    if visibility not in {"raw", "filtered", "both"}:
        raise ValueError("visibility must be raw, filtered, or both")
    traces = []
    landmarks = frame["landmarks"]
    if visibility in {"raw", "both"}:
        traces.append({
            "kind": "raw",
            "segments": build_skeleton_segments(landmarks, prefix="raw"),
        })
    if visibility in {"filtered", "both"}:
        traces.append({
            "kind": "filtered",
            "segments": build_skeleton_segments(landmarks, prefix="filt"),
        })
    return traces


def build_logged_playback_data(
    df: pd.DataFrame,
    visibility: str = "both",
) -> List[Dict[str, Any]]:
    return [
        {
            "frameNo": frame["frameNo"],
            "traces": build_logged_skeleton_trace_structures(frame, visibility),
        }
        for frame in group_frames(df)
    ]


def _result_to_sim_values(result: FilterResult, axis: str) -> Dict[str, float]:
    return {
        f"simFilt{axis}": result.position,
        f"simVel{axis}": result.velocity,
        f"simAccel{axis}": result.acceleration,
        f"simCovPosPos{axis}": result.covPosPos,
        f"simCovPosVel{axis}": result.covPosVel,
        f"simCovPosAccel{axis}": result.covPosAccel,
        f"simCovVelPos{axis}": result.covVelPos,
        f"simCovVelVel{axis}": result.covVelVel,
        f"simCovVelAccel{axis}": result.covVelAccel,
        f"simCovAccelPos{axis}": result.covPosAccel,
        f"simCovAccelVel{axis}": result.covVelAccel,
        f"simCovAccelAccel{axis}": result.covAccelAccel,
        f"simInnovationNormSquared{axis}": result.innovationNormSquared,
    }


def _state_to_sim_values(state: FilterAxisState, axis: str) -> Dict[str, float]:
    return {
        f"simFilt{axis}": state.position,
        f"simVel{axis}": state.velocity,
        f"simAccel{axis}": state.acceleration,
        f"simCovPosPos{axis}": state.covPosPos,
        f"simCovPosVel{axis}": state.covPosVel,
        f"simCovPosAccel{axis}": state.covPosAccel,
        f"simCovVelPos{axis}": state.covVelPos,
        f"simCovVelVel{axis}": state.covVelVel,
        f"simCovVelAccel{axis}": state.covVelAccel,
        f"simCovAccelPos{axis}": state.covAccelPos,
        f"simCovAccelVel{axis}": state.covAccelVel,
        f"simCovAccelAccel{axis}": state.covAccelAccel,
        f"simInnovationNormSquared{axis}": state.innovationNormSquared,
    }


def _filter_version() -> str:
    try:
        from world_space_kalman_filter_private import FILTER_VERSION
        return str(FILTER_VERSION)
    except Exception:
        return "missing-private-filter"


def _settings_signature(settings: PoseDetectionSettings) -> str:
    payload = json.dumps(asdict(settings), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _simulation_data_signature(df: pd.DataFrame) -> str:
    cols = ["frameNo", "landmark", "variant", "rawX", "rawY", "rawZ"]
    available = [col for col in cols if col in df.columns]
    payload = (
        df[available]
        .sort_values(["frameNo", "landmark"])
        .to_csv(index=False)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_cache_db_for_report(out_path: Path) -> Path:
    out_path = Path(out_path)
    return out_path.with_name(f"{out_path.stem}_world_filter_cache.sqlite")


def _init_simulation_cache(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            data_hash TEXT PRIMARY KEY,
            row_count INTEGER NOT NULL,
            frame_min INTEGER,
            frame_max INTEGER,
            landmarks TEXT NOT NULL,
            variants TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kalman_filter_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hash TEXT NOT NULL,
            settings_hash TEXT NOT NULL,
            filter_version TEXT NOT NULL,
            fps REAL NOT NULL,
            backend TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            output_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(data_hash, settings_hash, filter_version, fps),
            FOREIGN KEY(data_hash) REFERENCES datasets(data_hash)
        )
    """)
    conn.commit()


def _record_dataset_cache_metadata(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    data_hash: str,
) -> None:
    frame_min = int(df["frameNo"].min()) if len(df) else None
    frame_max = int(df["frameNo"].max()) if len(df) else None
    landmarks = json.dumps(sorted(df["landmark"].astype(str).unique().tolist()))
    variants = json.dumps(sorted(df["variant"].astype(str).unique().tolist()))
    conn.execute(
        """
        INSERT OR IGNORE INTO datasets
        (data_hash, row_count, frame_min, frame_max, landmarks, variants)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (data_hash, int(len(df)), frame_min, frame_max, landmarks, variants),
    )


def _load_cached_simulation(
    cache_db: Path,
    data_hash: str,
    settings_hash: str,
    filter_version: str,
    fps: float,
) -> Optional[pd.DataFrame]:
    if not cache_db.exists():
        return None
    with sqlite3.connect(cache_db) as conn:
        _init_simulation_cache(conn)
        row = conn.execute(
            """
            SELECT output_json FROM kalman_filter_outputs
            WHERE data_hash = ? AND settings_hash = ?
              AND filter_version = ? AND fps = ?
            """,
            (data_hash, settings_hash, filter_version, float(fps)),
        ).fetchone()
    if not row:
        return None
    return pd.read_json(io.StringIO(row[0]), orient="split")


def _save_cached_simulation(
    cache_db: Path,
    source_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    data_hash: str,
    settings_hash: str,
    filter_version: str,
    fps: float,
    backend: str,
) -> None:
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    output_json = sim_df.to_json(orient="split", double_precision=15)
    with sqlite3.connect(cache_db) as conn:
        _init_simulation_cache(conn)
        _record_dataset_cache_metadata(conn, source_df, data_hash)
        conn.execute(
            """
            INSERT OR REPLACE INTO kalman_filter_outputs
            (data_hash, settings_hash, filter_version, fps, backend, row_count, output_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data_hash,
                settings_hash,
                filter_version,
                float(fps),
                backend,
                int(len(sim_df)),
                output_json,
            ),
        )
        conn.commit()


def _simulate_world_space_filter_sequential(
    df: pd.DataFrame,
    settings: Optional[PoseDetectionSettings] = None,
    fps: float = 30.0,
    filter_obj: Optional[WorldSpaceKalmanFilter] = None,
) -> pd.DataFrame:
    settings = settings or PoseDetectionSettings()
    filter_obj = filter_obj or WorldSpaceKalmanFilter(settings)
    state_by_axis: Dict[Tuple[str, str], FilterAxisState] = {}
    previous_frame_by_axis: Dict[Tuple[str, str], int] = {}
    rows = []

    for _, row in df.sort_values(["frameNo", "landmark"]).iterrows():
        out = row.to_dict()
        frame_no = int(row["frameNo"])
        landmark = str(row["landmark"])
        for axis in AXES:
            key = (landmark, axis)
            measured_pos = float(row[f"raw{axis}"])
            previous_frame = previous_frame_by_axis.get(key)
            dt = compute_dt(previous_frame, frame_no, fps)
            out[f"simDt{axis}"] = dt

            if key not in state_by_axis:
                state = FilterAxisState.initialized(measured_pos)
                state_by_axis[key] = state
                out.update(_state_to_sim_values(state, axis))
            else:
                state = state_by_axis[key]
                result = filter_obj.filter_axis(
                    measuredPos=measured_pos,
                    dt=dt,
                    previousPos=state.position,
                    previousVel=state.velocity,
                    previousAccel=state.acceleration,
                    previousCovPosPos=state.covPosPos,
                    previousCovPosVel=state.covPosVel,
                    previousCovPosAccel=state.covPosAccel,
                    previousCovVelPos=state.covVelPos,
                    previousCovVelVel=state.covVelVel,
                    previousCovVelAccel=state.covVelAccel,
                    previousCovAccelPos=state.covAccelPos,
                    previousCovAccelVel=state.covAccelVel,
                    previousCovAccelAccel=state.covAccelAccel,
                    baseProcessNoisePos=settings.Q,
                    baseProcessNoiseVel=settings.Q,
                    baseProcessNoiseAccel=settings.baseProcessNoiseAccel,
                    baseMeasurementNoise=settings.R,
                    adaptiveQScalingFactor=settings.adaptiveQScalingFactor,
                    dynamicRInnovationSlope=settings.dynamicRInnovationSlope,
                    dynamicRInnovationThreshold=settings.dynamicRInnovationThreshold,
                    lastInnovationNormSquared=state.innovationNormSquared,
                    positionAlphaMin=settings.positionAlphaMin,
                    positionAlphaMax=settings.positionAlphaMax,
                    positionAlphaDecay=settings.positionAlphaDecay,
                    velocityAlphaMin=settings.velocityAlphaMin,
                    velocityAlphaMax=settings.velocityAlphaMax,
                    velocityAlphaMidpoint=settings.velocityAlphaMidpoint,
                    velocityAlphaSlope=settings.velocityAlphaSlope,
                    accelerationAlphaMin=settings.accelerationAlphaMin,
                    accelerationAlphaMax=settings.accelerationAlphaMax,
                    accelerationAlphaMidpoint=settings.accelerationAlphaMidpoint,
                    accelerationAlphaSlope=settings.accelerationAlphaSlope,
                    accelerationAlphaDecay=settings.accelerationAlphaDecay,
                    accelerationDecayRate=settings.accelerationDecayRate,
                    smoothingDecayTimeConstant=settings.smoothingDecayTimeConstant,
                    updateDecayTimeConstant=settings.updateDecayTimeConstant,
                    accelerationBoundsDecayFactor=settings.accelerationBoundsDecayFactor,
                    accelerationInnovationContributionFactor=settings.accelerationInnovationContributionFactor,
                    accelerationBoundsThreshold=settings.accelerationBoundsThreshold,
                    accelerationSaturationThreshold=settings.accelerationSaturationThreshold,
                    accelerationSaturationNoiseReduction=settings.accelerationSaturationNoiseReduction,
                    accelerationAdaptiveScalingFactor=settings.accelerationAdaptiveScalingFactor,
                    gainAttenuationRate=settings.gainAttenuationRate,
                    positionLockThreshold=settings.positionLockThreshold,
                    bypassCoercion=False,
                )
                state_by_axis[key] = FilterAxisState.from_result(result)
                out.update(_result_to_sim_values(result, axis))

            previous_frame_by_axis[key] = frame_no
        rows.append(out)

    return pd.DataFrame(rows)


def _simulate_landmark_worker(args):
    chunk_json, settings_values, fps = args
    chunk = pd.read_json(io.StringIO(chunk_json), orient="split")
    settings = PoseDetectionSettings(**settings_values)
    return _simulate_world_space_filter_sequential(chunk, settings, fps).to_json(
        orient="split",
        double_precision=15,
    )


def simulate_world_space_filter(
    df: pd.DataFrame,
    settings: Optional[PoseDetectionSettings] = None,
    fps: float = 30.0,
    filter_obj: Optional[WorldSpaceKalmanFilter] = None,
    parallel: bool = False,
    workers: Optional[int] = None,
    backend: str = "thread",
) -> pd.DataFrame:
    settings = settings or PoseDetectionSettings()
    if filter_obj is not None or not parallel:
        return _simulate_world_space_filter_sequential(df, settings, fps, filter_obj)

    groups = [
        group.sort_values(["frameNo", "landmark"]).reset_index(drop=True)
        for _, group in df.groupby("landmark", sort=True)
    ]
    if len(groups) <= 1:
        return _simulate_world_space_filter_sequential(df, settings, fps)

    worker_count = workers or min(4, len(groups))
    worker_count = max(1, min(int(worker_count), len(groups)))
    executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    settings_values = asdict(settings)
    tasks = [
        (group.to_json(orient="split", double_precision=15), settings_values, fps)
        for group in groups
    ]
    with executor_cls(max_workers=worker_count) as executor:
        parts = [
            pd.read_json(io.StringIO(result), orient="split")
            for result in executor.map(_simulate_landmark_worker, tasks)
        ]
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["frameNo", "landmark"])
        .reset_index(drop=True)
    )


def simulate_world_space_filter_cached(
    df: pd.DataFrame,
    settings: Optional[PoseDetectionSettings] = None,
    fps: float = 30.0,
    cache_db: Optional[Path] = None,
    parallel: bool = False,
    workers: Optional[int] = None,
    backend: str = "thread",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    settings = settings or PoseDetectionSettings()
    if cache_db is None:
        return simulate_world_space_filter(
            df, settings=settings, fps=fps,
            parallel=parallel, workers=workers, backend=backend,
        )

    cache_db = Path(cache_db)
    data_hash = _simulation_data_signature(df)
    settings_hash = _settings_signature(settings)
    filter_version = _filter_version()
    cached = _load_cached_simulation(
        cache_db, data_hash, settings_hash, filter_version, fps,
    )
    if cached is not None:
        message = f"simulation cache hit: {cache_db.name} {data_hash[:8]}"
        print(f"[info] {message}")
        _emit_progress(progress_callback, message)
        return cached

    message = f"simulation cache miss: {cache_db.name} {data_hash[:8]}"
    print(f"[info] {message}")
    _emit_progress(progress_callback, message)
    _emit_progress(
        progress_callback,
        f"Running Kalman simulation with {backend} backend"
        f"{' in parallel' if parallel else ' sequentially'}...",
    )
    sim_df = simulate_world_space_filter(
        df, settings=settings, fps=fps,
        parallel=parallel, workers=workers, backend=backend,
    )
    _save_cached_simulation(
        cache_db, df, sim_df, data_hash, settings_hash,
        filter_version, fps, backend,
    )
    _emit_progress(progress_callback, f"Stored simulated filter output in {cache_db}")
    return sim_df


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def enrich(df, iqr_k, min_delta):
    df = df.copy()
    for ax in AXES:
        df[f"res{ax}"]    = df[f"filt{ax}"] - df[f"raw{ax}"]
        df[f"absRes{ax}"] = df[f"res{ax}"].abs()

    def _per_group(g, lm=None, var=None):
        g = g.sort_values("frameNo").copy()
        if "landmark" not in g.columns or "variant" not in g.columns:
            g["landmark"] = lm
            g["variant"] = var
        any_flag = pd.Series(False, index=g.index)
        for ax in AXES:
            d_raw  = g[f"raw{ax}"].diff().abs()
            d_filt = g[f"filt{ax}"].diff().abs()
            g[f"dRaw{ax}"], g[f"dFilt{ax}"] = d_raw, d_filt
            q1, q3 = d_raw.quantile(0.25), d_raw.quantile(0.75)
            thr = q3 + iqr_k * (q3 - q1)
            flag = (d_raw > thr) & (d_raw > min_delta) & d_raw.notna()
            g[f"outlier{ax}"] = flag; any_flag |= flag
        g["outlier"] = any_flag
        return g

    grouped = [
        _per_group(g, lm=lm, var=var)
        for (lm, var), g in df.groupby(["landmark", "variant"], group_keys=False)
    ]
    return (pd.concat(grouped, ignore_index=True)
              .sort_values(["landmark", "variant", "frameNo"])
              .reset_index(drop=True))


def summarize(df):
    recs = []
    for (lm, var), g in df.groupby(["landmark", "variant"]):
        rec = {"landmark": lm, "variant": var,
               "tag": g["tag"].iloc[0], "alpha": g["alpha"].iloc[0],
               "nFrames": len(g), "nOutliers": int(g["outlier"].sum())}
        jit = []
        for ax in AXES:
            ar = g[f"absRes{ax}"]
            rec[f"std{ax}"]  = g[f"res{ax}"].std()
            rec[f"mean{ax}"] = ar.mean()
            rec[f"p25{ax}"]  = ar.quantile(0.25)
            rec[f"p50{ax}"]  = ar.quantile(0.50)
            rec[f"p75{ax}"]  = ar.quantile(0.75)
            sr, sf = g[f"dRaw{ax}"].std(), g[f"dFilt{ax}"].std()
            if sr and sr > 0:
                jit.append(max(0.0, 1.0 - sf / sr))
        for k in ("std", "mean", "p25", "p50", "p75"):
            rec[f"{k}Avg"] = sum(rec[f"{k}{ax}"] for ax in AXES) / 3.0
        rec["jitterReduction"] = sum(jit) / len(jit) if jit else 0.0
        recs.append(rec)
    out = pd.DataFrame(recs)
    out["rankInLandmark"] = out.groupby("landmark")["p50Avg"].rank(method="min")
    out["bestInLandmark"] = out["rankInLandmark"] == 1.0
    return out


# ---------------------------------------------------------------------------
# 2D plotting
# ---------------------------------------------------------------------------
AXIS_COLORS = {"X": "#1f77b4", "Y": "#ff7f0e", "Z": "#2ca02c"}


def _line_traces(g, landmark, showlegend):
    frames = g["frameNo"].values; out = []
    for ax in AXES:
        out.append(go.Scatter(x=frames, y=g[f"raw{ax}"],
            name=f"raw {ax}", legendgroup=f"raw{ax}", showlegend=showlegend,
            mode="lines", line=dict(color=AXIS_COLORS[ax], width=1.0), opacity=0.35,
            hovertemplate=f"{landmark} raw{ax}: %{{y:.4f}}<extra></extra>"))
        out.append(go.Scatter(x=frames, y=g[f"filt{ax}"],
            name=f"filt {ax}", legendgroup=f"filt{ax}", showlegend=showlegend,
            mode="lines", line=dict(color=AXIS_COLORS[ax], width=2.2),
            hovertemplate=f"{landmark} filt{ax}: %{{y:.4f}}<extra></extra>"))
    return out


def _outlier_traces(g, landmark, showlegend):
    out, first = [], showlegend
    for ax in AXES:
        mask = g[f"outlier{ax}"]
        if not mask.any():
            continue
        out.append(go.Scatter(x=g.loc[mask, "frameNo"], y=g.loc[mask, f"raw{ax}"],
            name="outlier", legendgroup="outlier", showlegend=first,
            mode="markers", marker=dict(color="red", size=7, symbol="x"),
            hovertemplate=f"{landmark} outlier raw{ax} @ f%{{x}}: %{{y:.4f}}<extra></extra>"))
        first = False
    return out


def build_group_figure(df, pairs):
    n = len(pairs); ncols = min(4, n); nrows = math.ceil(n / ncols)
    vspace = min(0.05, 0.5 / max(nrows - 1, 1))
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[f"{lm} â€” {v}" for lm, v in pairs],
                        shared_xaxes=True,
                        horizontal_spacing=0.04, vertical_spacing=vspace)
    for idx, (lm, var) in enumerate(pairs):
        r, c = divmod(idx, ncols)
        g = df[(df["landmark"] == lm) & (df["variant"] == var)]
        for tr in _line_traces(g, lm, showlegend=(idx == 0)):
            fig.add_trace(tr, row=r + 1, col=c + 1)
        for tr in _outlier_traces(g, lm, showlegend=(idx == 0)):
            fig.add_trace(tr, row=r + 1, col=c + 1)
    fig.update_layout(title="Group â€” all (landmark, variant) pairs",
                      height=max(280 * nrows, 600), hovermode="x unified",
                      margin=dict(t=80),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1.0))
    fig.update_annotations(font_size=10)
    return fig


def build_detail_figure(g, landmark, variant):
    fig = go.Figure()
    for tr in _line_traces(g, landmark, showlegend=True):
        fig.add_trace(tr)
    for tr in _outlier_traces(g, landmark, showlegend=True):
        fig.add_trace(tr)
    fig.update_layout(title=f"{landmark} â€” {variant}",
                      xaxis_title="frameNo", yaxis_title="world coord (m)",
                      hovermode="x unified", height=620,
                      margin=dict(t=60, b=40),
                      legend=dict(orientation="h", y=-0.18))
    return fig


# ---------------------------------------------------------------------------
# 3D plotting
# ---------------------------------------------------------------------------
_CAM_PRESETS = {
    "Free":          dict(eye=dict(x=1.6, y=1.6, z=1.2), up=dict(x=0, y=0, z=1)),
    "Along X â†’ YZ":  dict(eye=dict(x=2.2, y=0,   z=0  ), up=dict(x=0, y=0, z=1)),
    "Along Y â†’ XZ":  dict(eye=dict(x=0,   y=2.2, z=0  ), up=dict(x=0, y=0, z=1)),
    "Along Z â†’ XY":  dict(eye=dict(x=0,   y=0,   z=2.2), up=dict(x=0, y=1, z=0)),
}


def build_3d_figure(g, landmark, variant):
    g = g.sort_values("frameNo")
    frames = g["frameNo"].values
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=g["rawX"], y=g["rawY"], z=g["rawZ"],
        mode="lines", name="raw",
        line=dict(color="gray", width=3), opacity=0.35, customdata=frames,
        hovertemplate="raw f%{customdata}<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>"
                      "Z=%{z:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter3d(
        x=g["filtX"], y=g["filtY"], z=g["filtZ"],
        mode="lines", name="filt",
        line=dict(color=frames, colorscale="Viridis", width=6,
                  colorbar=dict(title="frame", thickness=14, len=0.6, x=1.0)),
        customdata=frames,
        hovertemplate="filt f%{customdata}<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>"
                      "Z=%{z:.4f}<extra></extra>"))
    mask = g["outlier"]
    if mask.any():
        fig.add_trace(go.Scatter3d(
            x=g.loc[mask, "rawX"], y=g.loc[mask, "rawY"], z=g.loc[mask, "rawZ"],
            mode="markers", name="outlier",
            marker=dict(color="red", size=4, symbol="x"),
            customdata=g.loc[mask, "frameNo"],
            hovertemplate="outlier f%{customdata}<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>"
                          "Z=%{z:.4f}<extra></extra>"))
    buttons = [dict(label=k, method="relayout", args=[{"scene.camera": v}])
               for k, v in _CAM_PRESETS.items()]
    fig.update_layout(
        title=f"{landmark} â€” {variant}",
        height=620, margin=dict(t=60, b=10, l=10, r=10),
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                   aspectmode="data", camera=_CAM_PRESETS["Free"]),
        updatemenus=[dict(type="dropdown", buttons=buttons, direction="down",
                          x=0.0, xanchor="left", y=1.06, yanchor="bottom")],
        legend=dict(x=0.0, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.6)"))
    return fig


# ---------------------------------------------------------------------------
# HTML â€” lazy mount/purge
# ---------------------------------------------------------------------------
def _segment_arrays(segments):
    x, y, z = [], [], []
    for segment in segments:
        start = _skeleton_display_coords(
            segment["x"][0], segment["y"][0], segment["z"][0])
        end = _skeleton_display_coords(
            segment["x"][1], segment["y"][1], segment["z"][1])
        x.extend([start[0], end[0], None])
        y.extend([start[1], end[1], None])
        z.extend([start[2], end[2], None])
    return x, y, z


def _skeleton_display_coords(world_x, world_y, world_z):
    """Map MediaPipe world coordinates into a Y-up skeleton display space."""
    return float(world_x), -float(world_y), float(world_z)


def _point_arrays(landmarks, prefix):
    x, y, z, labels = [], [], [], []
    for name, row in sorted(landmarks.items()):
        keys = [f"{prefix}{axis}" for axis in AXES]
        if not all(key in row for key in keys):
            continue
        display_x, display_y, display_z = _skeleton_display_coords(
            _coord(row, prefix, "X"),
            _coord(row, prefix, "Y"),
            _coord(row, prefix, "Z"),
        )
        x.append(display_x)
        y.append(display_y)
        z.append(display_z)
        labels.append(name)
    return x, y, z, labels


def _skeleton_plot_payload(landmarks, prefix):
    line_x, line_y, line_z = _segment_arrays(build_skeleton_segments(landmarks, prefix))
    point_x, point_y, point_z, point_labels = _point_arrays(landmarks, prefix)
    return {
        "line": dict(x=line_x, y=line_y, z=line_z),
        "points": dict(x=point_x, y=point_y, z=point_z, text=point_labels),
    }


def _skeleton_frame_maps(df):
    return {frame["frameNo"]: frame["landmarks"] for frame in group_frames(df)}


def _stable_axis_range(values, pad_fraction=0.08):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return [-1.0, 1.0]
    lo, hi = min(finite), max(finite)
    span = hi - lo
    if span <= 1e-9:
        pad = max(abs(lo) * 0.1, 0.1)
    else:
        pad = span * pad_fraction
    return [lo - pad, hi + pad]


def _skeleton_scene_ranges(frame_payloads):
    values = {"x": [], "y": [], "z": []}
    for payload_tuple in frame_payloads.values():
        for payload in payload_tuple:
            for part in ("line", "points"):
                for axis in values:
                    values[axis].extend(payload[part].get(axis, []))
    ranges = {axis: _stable_axis_range(axis_values)
              for axis, axis_values in values.items()}
    spans = {axis: max(ranges[axis][1] - ranges[axis][0], 1e-9)
             for axis in ranges}
    max_span = max(spans.values())
    return ranges, {axis: spans[axis] / max_span for axis in spans}


def build_skeleton_figure(
    df: pd.DataFrame,
    variant: str,
    fps: float = 30.0,
    settings: Optional[PoseDetectionSettings] = None,
    cache_db: Optional[Path] = None,
    simulation_parallel: bool = True,
    simulation_workers: Optional[int] = None,
    simulation_backend: str = "thread",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> go.Figure:
    settings = settings or PoseDetectionSettings()
    df = df.sort_values(["frameNo", "landmark"]).reset_index(drop=True)
    frame_numbers = sorted_frame_numbers(df)
    fig = go.Figure()
    if not frame_numbers:
        fig.update_layout(title=f"Skeleton - {variant}", height=900)
        return fig

    logged_frames = _skeleton_frame_maps(df)
    _emit_progress(progress_callback, f"Preparing simulated skeleton data for {variant}...")
    sim_frames = _skeleton_frame_maps(simulate_world_space_filter_cached(
        df,
        settings=settings,
        fps=fps,
        cache_db=cache_db,
        parallel=simulation_parallel,
        workers=simulation_workers,
        backend=simulation_backend,
        progress_callback=progress_callback,
    ))
    first_frame = frame_numbers[0]

    def payloads(frame_no):
        logged = logged_frames[frame_no]
        simulated = sim_frames[frame_no]
        return (
            _skeleton_plot_payload(logged, "raw"),
            _skeleton_plot_payload(logged, "filt"),
            _skeleton_plot_payload(simulated, "simFilt"),
        )

    raw_payload, logged_payload, sim_payload = payloads(first_frame)
    line_style = dict(width=7)
    point_style = dict(size=4)
    fig.add_traces([
        go.Scatter3d(
            **raw_payload["line"], mode="lines", name="raw skeleton",
            legendgroup="raw", line=dict(color=SKELETON_RAW_COLOR, **line_style),
            opacity=0.48, hoverinfo="skip",
        ),
        go.Scatter3d(
            **raw_payload["points"], mode="markers+text", name="raw landmarks",
            legendgroup="raw", showlegend=False,
            marker=dict(color=SKELETON_RAW_COLOR, **point_style), textposition="top center",
            hovertemplate="%{text}<br>raw X=%{x:.4f}<br>height Y=%{y:.4f}<br>Z=%{z:.4f}<extra></extra>",
        ),
        go.Scatter3d(
            **logged_payload["line"], mode="lines", name="logged filtered skeleton",
            legendgroup="logged-filtered",
            line=dict(color=SKELETON_LOGGED_FILTERED_COLOR, **line_style),
            hoverinfo="skip",
        ),
        go.Scatter3d(
            **logged_payload["points"], mode="markers+text",
            name="logged filtered landmarks", legendgroup="logged-filtered",
            showlegend=False,
            marker=dict(color=SKELETON_LOGGED_FILTERED_COLOR, **point_style),
            textposition="bottom center",
            hovertemplate="%{text}<br>logged filt X=%{x:.4f}<br>height Y=%{y:.4f}<br>Z=%{z:.4f}<extra></extra>",
        ),
        go.Scatter3d(
            **sim_payload["line"], mode="lines", name="simulated filtered skeleton",
            legendgroup="sim-filtered", visible=False,
            line=dict(color=SKELETON_SIM_FILTERED_COLOR, **line_style), hoverinfo="skip",
        ),
        go.Scatter3d(
            **sim_payload["points"], mode="markers+text",
            name="simulated filtered landmarks", legendgroup="sim-filtered",
            showlegend=False, visible=False,
            marker=dict(color=SKELETON_SIM_FILTERED_COLOR, **point_style),
            textposition="bottom center",
            hovertemplate="%{text}<br>sim filt X=%{x:.4f}<br>height Y=%{y:.4f}<br>Z=%{z:.4f}<extra></extra>",
        ),
    ])

    frame_data = {}
    for frame_no in frame_numbers:
        frame_data[frame_no] = payloads(frame_no)
    scene_ranges, aspect_ratio = _skeleton_scene_ranges(frame_data)
    fig.frames = [
        go.Frame(
            name=str(frame_no),
            traces=[0, 1, 2, 3, 4, 5],
            data=[
                go.Scatter3d(**frame_data[frame_no][0]["line"]),
                go.Scatter3d(**frame_data[frame_no][0]["points"]),
                go.Scatter3d(**frame_data[frame_no][1]["line"]),
                go.Scatter3d(**frame_data[frame_no][1]["points"]),
                go.Scatter3d(**frame_data[frame_no][2]["line"]),
                go.Scatter3d(**frame_data[frame_no][2]["points"]),
            ],
        )
        for frame_no in frame_numbers
    ]

    frame_ms = max(int(1000.0 / fps), 1)
    fig.update_layout(
        title=f"Skeleton - {variant}",
        height=900,
        margin=dict(t=42, b=4, l=4, r=4),
        scene=dict(
            xaxis=dict(title="X floor (m)", range=scene_ranges["x"], autorange=False),
            yaxis=dict(title="Y height (m)", range=scene_ranges["y"], autorange=False),
            zaxis=dict(title="Z depth (m)", range=scene_ranges["z"], autorange=False),
            aspectmode="manual",
            aspectratio=dict(x=aspect_ratio["x"], y=aspect_ratio["y"],
                             z=aspect_ratio["z"]),
            camera=dict(eye=dict(x=1.65, y=0.85, z=1.45),
                        up=dict(x=0, y=1, z=0),
                        projection=dict(type="orthographic")),
            dragmode="turntable",
            uirevision="skeleton-fixed-scene",
        ),
        uirevision="skeleton-fixed-scene",
        legend=dict(x=0.0, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.6)"),
    )
    return fig


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


def _quick_glance_html(row):
    star = " â˜…" if row.get("bestInLandmark", False) else ""
    cells = [("Î±", f"{row['alpha']:g}{star}"), ("frames", int(row["nFrames"])),
             ("|res| p25", _fmt(row["p25Avg"])), ("p50", _fmt(row["p50Avg"])),
             ("p75", _fmt(row["p75Avg"])), ("mean", _fmt(row["meanAvg"])),
             ("std", _fmt(row["stdAvg"])),
             ("jitterâ†“", f"{row['jitterReduction']*100:.1f}%"),
             ("outlier frames", int(row["nOutliers"]))]
    head = "<tr>" + "".join(f"<th>{k}</th>" for k, _ in cells) + "</tr>"
    body = "<tr>" + "".join(f"<td>{v}</td>" for _, v in cells) + "</tr>"
    return f"<table class='quick'>{head}{body}</table>"


def _per_axis_table_html(row):
    head = "<tr><th></th>" + "".join(f"<th>{ax}</th>" for ax in AXES) + "<th>avg(XYZ)</th></tr>"
    body = ""
    for label, key in [("std(res)", "std"), ("mean |res|", "mean"),
                       ("p25 |res|", "p25"), ("p50 |res|", "p50"),
                       ("p75 |res|", "p75")]:
        body += f"<tr><td><b>{label}</b></td>"
        for ax in AXES:
            body += f"<td>{_fmt(row[key + ax], 5)}</td>"
        body += f"<td>{_fmt(row[key + 'Avg'], 5)}</td></tr>"
    return f"<table class='stats'>{head}{body}</table>"


def _ranking_table_html(summary):
    s = summary.sort_values("p50Avg").reset_index(drop=True)
    head = ("<tr><th>#</th><th>landmark</th><th>variant</th>"
            "<th>p50 |res|</th><th>p75 |res|</th><th>jitterâ†“</th>"
            "<th>outliers</th><th>frames</th></tr>")
    rows = []
    for i, r in s.iterrows():
        star = " â˜…" if r["bestInLandmark"] else ""
        rows.append(
            f"<tr><td>{i+1}</td>"
            f"<td><a href='#' onclick=\"setLandmark('{r['landmark']}');return false;\">"
            f"{r['landmark']}</a></td>"
            f"<td style='text-align:left'>{r['variant']}{star}</td>"
            f"<td>{_fmt(r['p50Avg'])}</td><td>{_fmt(r['p75Avg'])}</td>"
            f"<td>{r['jitterReduction']*100:.1f}%</td>"
            f"<td>{int(r['nOutliers'])}</td><td>{int(r['nFrames'])}</td></tr>")
    return ("<details class='ranking'><summary><b>Ranking â€” all (landmark, variant) "
            "by p50 |res| (â˜… = best variant for that landmark)</b></summary>"
            f"<table class='stats'>{head}{''.join(rows)}</table></details>")


def _plotlyjs_cdn_tag():
    try:
        from plotly.offline.offline import get_plotlyjs_version
        ver = get_plotlyjs_version()
    except Exception:
        ver = "2.35.2"
    return (f'<script src="https://cdn.plot.ly/plotly-{ver}.min.js" '
            f'charset="utf-8"></script>')


def _lazy_plot_html(fig, plot_id, keep=False, extra_class=""):
    """Ship figure as JSON + empty placeholder div; mounted on demand in JS."""
    h = int(fig.layout.height or 450)
    fig_json = fig.to_json().replace("</", "<\\/")   # script-tag safety
    keep_attr = ' data-keep="1"' if keep else ""
    classes = f"lazy-plot {extra_class}".strip()
    return (f'<div id="plot-{plot_id}" class="{classes}" data-figid="{plot_id}"'
            f'{keep_attr} style="min-height:{h}px"></div>'
            f'<script type="application/json" id="figdata-{plot_id}">'
            f'{fig_json}</script>')


def _settings_inputs_html(settings):
    values = asdict(settings)
    controls = []
    for f in fields(PoseDetectionSettings):
        name = f.name
        step = "1" if name == "numPreviousLocations" else "any"
        controls.append(
            "<label class='coef-field' "
            f"for='coef-{name}'><span>{name}</span>"
            f"<input id='coef-{name}' data-coef='{name}' type='number' "
            f"step='{step}' value='{values[name]}'></label>"
        )
    return (
        "<details class='coefficients'><summary><b>Filter coefficients</b></summary>"
        "<div class='coef-grid'>" + "".join(controls) + "</div></details>"
    )


HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"/>
<title>WORLD Filter Review</title>
<style>
  body { font-family: sans-serif; margin: 0; padding: 12px; }
  .toggle-bar { position: sticky; top: 0; background: #fff; padding: 8px 0;
                border-bottom: 1px solid #ddd; z-index: 10;
                display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .toggle-bar button, .toggle-bar select {
      font-size: 14px; padding: 6px 12px; cursor: pointer; }
  .toggle-bar button.active { background: #1f77b4; color: #fff; border: none; }
  .nav { display: none; gap: 4px; align-items: center; }
  .nav.active { display: inline-flex; }
  .nav button { padding: 6px 10px; font-size: 16px; }
  .view { display: none; } .view.active { display: block; }
  .lm-block { display: none; } .lm-block.active { display: block; }
  .alpha-block { border: 1px solid #eee; border-radius: 6px;
                 margin: 14px 0; padding: 8px 8px 14px 8px; }
  .lazy-plot { width: 100%; }
  table.stats, table.quick { border-collapse: collapse; margin: 6px 4px; font-size: 12px; }
  table.stats th, table.stats td,
  table.quick th, table.quick td { border: 1px solid #ccc; padding: 3px 8px; text-align: right; }
  table.stats th, table.quick th { background: #f3f3f3; }
  table.quick td, table.quick th { font-size: 13px; }
  .stats-caption { font-size: 12px; color: #555; margin: 2px 4px; }
  .params { font-size: 12px; color: #666; margin-left: auto; }
  details.ranking { margin: 10px 0 18px 0; }
  details.ranking summary { cursor: pointer; font-size: 14px; padding: 6px 0; }
  details.coefficients { margin: 10px 0; }
  details.coefficients summary { cursor: pointer; font-size: 14px; padding: 6px 0; }
  .coef-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
               gap: 6px 10px; margin: 8px 0 12px; }
  .coef-field { display: flex; align-items: center; gap: 6px; font-size: 12px; }
  .coef-field span { flex: 1 1 auto; overflow-wrap: anywhere; }
  .coef-field input { width: 92px; padding: 3px 5px; }
  .skeleton-controls { position: sticky; top: 50px; z-index: 9;
                       display: flex; align-items: center; gap: 10px;
                       flex-wrap: wrap; margin: 10px 0 0; font-size: 14px;
                       background: #fff; border: 1px solid #ddd;
                       border-bottom: 0; border-radius: 6px 6px 0 0;
                       padding: 8px 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .skeleton-controls button, .skeleton-controls select {
      font-size: 14px; padding: 5px 10px; cursor: pointer; }
  .skeleton-controls label { display: inline-flex; align-items: center; gap: 4px; }
  .skeleton-controls input[type=range] { width: 150px; }
  .skeleton-key { display: inline-flex; align-items: center; gap: 5px;
                  font-size: 12px; color: #333; }
  .skeleton-key::before { content: ""; width: 18px; height: 4px;
                          border-radius: 999px; background: var(--key-color); }
  .skeleton-key.raw { --key-color: #374151; }
  .skeleton-key.logged { --key-color: #2563eb; }
  .skeleton-key.sim { --key-color: #f97316; }
  #view-skeleton.active { padding-bottom: 64px; }
  .skeleton-scrubber { position: fixed; left: 12px; right: 12px; bottom: 0;
                       z-index: 10; display: flex; align-items: center;
                       gap: 10px; background: #fff; border: 1px solid #ddd;
                       border-radius: 6px 6px 0 0; padding: 8px 10px;
                       box-shadow: 0 -1px 8px rgba(0,0,0,0.10);
                       font-size: 13px; }
  .skeleton-scrubber input[type=range] { flex: 1 1 auto; min-width: 160px; }
  .skeleton-scrubber span { white-space: nowrap; }
  .skeleton-variant-block { display: none; }
  .skeleton-variant-block.active { display: block; }
  .skeleton-variant-block { border: 1px solid #ddd; border-top: 0;
                            margin-top: 0; border-radius: 0 0 6px 6px; }
  .skeleton-variant-block .lazy-plot { min-height: 82vh !important; }
  .skeleton-variant-block h3 { margin: 8px 10px 0; font-size: 14px; }
  .skeleton-log {
      border: 1px solid #ddd; background: #fafafa; color: #333;
      font-family: Consolas, monospace; font-size: 12px; line-height: 1.35;
      min-height: 64px; max-height: 180px; overflow: auto; padding: 8px;
      white-space: pre-wrap; margin: 8px 0 12px; }
  .skeleton-status { color: #555; font-size: 12px; }
</style></head><body>
"""

HTML_TAIL_TMPL = """
<script>
var LANDMARKS = __LANDMARKS__; var lmIndex = 0;
var VIEWS = ['group','detail','3d','skeleton'];
var mounted = {};
var skeletonFrameNames = {};
var skeletonFramePayloads = {};
var skeletonPlaybackTimers = {};
var skeletonPlaybackIndex = {};
var skeletonViewStateByPlot = {};
var skeletonSubjectYawDegrees = 0;
var skeletonSubjectSpinTimer = null;
var PLOT_CFG = {responsive: true, displaylogo: false, scrollZoom: true};
var SKELETON_FRAME_MS = __SKELETON_FRAME_MS__;
var SKELETON_DEFAULT_CAMERA = {
  eye: {x: 1.65, y: 0.85, z: 1.45},
  up: {x: 0, y: 1, z: 0},
  projection: {type: 'orthographic'}
};

function skeletonLog(msg){
  var text = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  console.log('[skeleton]', msg);
  var panel = document.getElementById('skeleton-log');
  if (!panel) return;
  panel.textContent += text + '\\n';
  panel.scrollTop = panel.scrollHeight;
}
function clearSkeletonLog(){
  var panel = document.getElementById('skeleton-log');
  if (panel) panel.textContent = '';
  skeletonLog('log cleared');
}

function isLive(div){
  var v = div.closest('.view');
  if (!v || !v.classList.contains('active')) return false;
  var b = div.closest('.lm-block');
  if (b && !b.classList.contains('active')) return false;
  var skel = div.closest('.skeleton-variant-block');
  if (skel && !skel.classList.contains('active')) return false;
  return true;
}
function refreshMounts(){
  document.querySelectorAll('.lazy-plot').forEach(function(div){
    var id = div.dataset.figid, live = isLive(div);
    if (live && !mounted[id]) {
      try {
        var j = JSON.parse(document.getElementById('figdata-'+id).textContent);
        Plotly.newPlot(div, j.data, j.layout, PLOT_CFG);
        mounted[id] = true;
        if (div.classList.contains('skeleton-plot')) {
          skeletonFrameNames[id] = (j.frames || []).map(function(f){ return f.name; });
          skeletonFramePayloads[id] = (j.frames || []).map(function(f){ return f.data || []; });
          skeletonPlaybackIndex[id] = 0;
          skeletonLog('mounted '+id+' with '+skeletonFrameNames[id].length+' animation frames');
          attachSkeletonDebugHandlers(div);
          applySkeletonVisibilityToPlot(div);
          rememberSkeletonViewState(div);
          updateSkeletonFrameControls(div);
        }
      } catch(e){ console.error('mount '+id, e); }
    } else if (!live && mounted[id] && !div.dataset.keep) {
      if (div.classList.contains('skeleton-plot')) skeletonLog('purging '+id);
      try { Plotly.purge(div); } catch(e){}
      mounted[id] = false;
      delete skeletonFrameNames[id];
      delete skeletonFramePayloads[id];
      delete skeletonViewStateByPlot[id];
      stopSkeletonTimer(id);
    }
  });
}
function setView(v){
  VIEWS.forEach(function(k){
    document.getElementById('view-'+k).classList.toggle('active', v===k);
    document.getElementById('btn-'+k).classList.toggle('active', v===k);
  });
  document.getElementById('nav').classList.toggle('active', v!=='group' && v!=='skeleton');
  if (v === 'skeleton') skeletonLog('Skeleton view selected; refreshing mounts');
  refreshMounts();
  if (v === 'skeleton') updateSkeletonFrameControls(activeSkeletonPlot());
}
function setLandmark(lm){
  var i = LANDMARKS.indexOf(lm); if (i<0) return; lmIndex=i;
  document.querySelectorAll('.lm-block').forEach(function(el){
    el.classList.toggle('active', el.dataset.lm === lm); });
  document.getElementById('lm-select').value = lm;
  document.getElementById('lm-pos').textContent = (i+1)+' / '+LANDMARKS.length;
  window.scrollTo({top:0});
  refreshMounts();
}
function stepLandmark(d){ var n=LANDMARKS.length;
  lmIndex=((lmIndex+d)%n+n)%n; setLandmark(LANDMARKS[lmIndex]); }
function attachSkeletonDebugHandlers(div){
  if (div.dataset.skeletonDebugAttached) return;
  div.dataset.skeletonDebugAttached = '1';
  div.on('plotly_animatingframe', function(e){
    var frameName = e && e.name ? e.name : '?';
    var status = document.getElementById('skel-frame-status');
    if (status) status.textContent = 'Frame ' + frameName;
    skeletonLog(div.dataset.figid+' animating frame '+frameName);
  });
  div.on('plotly_animated', function(){
    skeletonLog(div.dataset.figid+' animation completed');
  });
  div.on('plotly_animationinterrupted', function(){
    skeletonLog(div.dataset.figid+' animation interrupted');
  });
  div.on('plotly_relayout', function(){
    rememberSkeletonViewState(div);
  });
  div.on('plotly_doubleclick', function(){
    window.setTimeout(function(){ rememberSkeletonViewState(div); }, 0);
  });
}
function applySkeletonVisibilityToPlot(div){
  var rawCtl = document.getElementById('skel-show-raw');
  var filtCtl = document.getElementById('skel-show-filtered');
  var modeCtl = document.getElementById('skel-mode');
  var showRaw = !rawCtl || rawCtl.checked;
  var showFiltered = !filtCtl || filtCtl.checked;
  var mode = modeCtl ? modeCtl.value : 'logged';
  var visible = [
    showRaw, showRaw,
    showFiltered && mode === 'logged', showFiltered && mode === 'logged',
    showFiltered && mode === 'simulation', showFiltered && mode === 'simulation'
  ];
  runWithSkeletonViewState(div, function(){
    return Plotly.restyle(div, {visible: visible});
  });
  var labelCtl = document.getElementById('skel-show-labels');
  var pointMode = labelCtl && labelCtl.checked ? 'markers+text' : 'markers';
  runWithSkeletonViewState(div, function(){
    return Plotly.restyle(div, {mode: [pointMode, pointMode, pointMode]}, [1, 3, 5]);
  });
  skeletonLog(div.dataset.figid+' visibility raw='+showRaw+' filtered='+showFiltered+' mode='+mode);
}
function setSkeletonVisibility(){
  skeletonLog('visibility control changed');
  document.querySelectorAll('.skeleton-plot').forEach(function(div){
    if (mounted[div.dataset.figid]) applySkeletonVisibilityToPlot(div);
  });
}
function setSkeletonVariant(value){
  skeletonLog('variant changed to '+value);
  pauseSkeletonPlots();
  document.querySelectorAll('.skeleton-variant-block').forEach(function(block){
    block.classList.toggle('active', block.dataset.variant === value);
  });
  refreshMounts();
  updateSkeletonFrameControls(activeSkeletonPlot());
}
function stopSkeletonTimer(id){
  if (skeletonPlaybackTimers[id]) {
    window.clearInterval(skeletonPlaybackTimers[id]);
    delete skeletonPlaybackTimers[id];
  }
}
function skeletonSpeedMultiplier(){
  var ctl = document.getElementById('skel-speed');
  var value = ctl ? Number(ctl.value) : 1;
  return Number.isFinite(value) && value > 0 ? value : 1;
}
function skeletonFrameStep(){
  var ctl = document.getElementById('skel-frame-step');
  var value = ctl ? Number(ctl.value) : 1;
  return Number.isFinite(value) && value > 0 ? Math.max(1, Math.floor(value)) : 1;
}
function skeletonFrameDelay(){
  return Math.max(8, Math.round(SKELETON_FRAME_MS / skeletonSpeedMultiplier()));
}
function skeletonSpaceMode(){
  var ctl = document.getElementById('skel-space');
  return ctl ? ctl.value : 'world';
}
function skeletonSubjectYawRadians(){
  return skeletonSubjectYawDegrees * Math.PI / 180;
}
function setSkeletonSpaceMode(){
  var div = activeSkeletonPlot();
  var mode = skeletonSpaceMode();
  skeletonLog('pose frame changed to '+mode);
  if (div) restyleSkeletonFrame(div, skeletonPlaybackIndex[div.dataset.figid] || 0, true);
}
function setSkeletonSubjectYaw(value){
  var number = Number(value);
  if (!Number.isFinite(number)) number = 0;
  skeletonSubjectYawDegrees = ((number % 360) + 360) % 360;
  if (skeletonSubjectYawDegrees > 180) skeletonSubjectYawDegrees -= 360;
  var slider = document.getElementById('skel-subject-yaw');
  var label = document.getElementById('skel-subject-yaw-value');
  if (slider) slider.value = skeletonSubjectYawDegrees;
  if (label) label.textContent = Math.round(skeletonSubjectYawDegrees) + ' deg';
  var div = activeSkeletonPlot();
  if (div) restyleSkeletonFrame(div, skeletonPlaybackIndex[div.dataset.figid] || 0, false);
}
function startSkeletonSubjectSpin(){
  stopSkeletonSubjectSpin();
  skeletonLog('subject twirl started');
  skeletonSubjectSpinTimer = window.setInterval(function(){
    setSkeletonSubjectYaw(skeletonSubjectYawDegrees + 3);
  }, 33);
}
function stopSkeletonSubjectSpin(){
  if (skeletonSubjectSpinTimer) {
    window.clearInterval(skeletonSubjectSpinTimer);
    skeletonSubjectSpinTimer = null;
    skeletonLog('subject twirl stopped');
  }
}
function restartSkeletonPlaybackIfRunning(){
  var anyRunning = Object.keys(skeletonPlaybackTimers).length > 0;
  if (anyRunning) playSkeletonPlots();
}
function updateSkeletonFrameControls(div){
  var slider = document.getElementById('skel-frame-slider');
  var count = document.getElementById('skel-frame-count');
  if (!slider) return;
  div = div || activeSkeletonPlot();
  if (!div) {
    slider.min = 0;
    slider.max = 0;
    slider.value = 0;
    slider.disabled = true;
    if (count) count.textContent = '0 / 0';
    return;
  }
  var id = div.dataset.figid;
  var names = skeletonFrameNames[id] || [];
  var index = skeletonPlaybackIndex[id] || 0;
  slider.disabled = names.length === 0;
  slider.min = 0;
  slider.max = Math.max(names.length - 1, 0);
  slider.value = Math.min(index, Number(slider.max));
  var frameName = names[index] || index;
  if (count) count.textContent = (names.length ? index + 1 : 0) +
    ' / ' + names.length + ' (frame ' + frameName + ')';
}
function seekSkeletonFrame(frameIndex){
  var div = activeSkeletonPlot();
  if (!div) {
    refreshMounts();
    div = activeSkeletonPlot();
  }
  if (!div) {
    skeletonLog('frame slider moved but no active skeleton plot is mounted');
    return;
  }
  var index = Number(frameIndex);
  if (!Number.isFinite(index)) index = 0;
  stopSkeletonTimer(div.dataset.figid);
  restyleSkeletonFrame(div, Math.round(index), true);
}
function pointByName(trace, name){
  if (!trace || !trace.text) return null;
  var labels = trace.text || [];
  for (var i = 0; i < labels.length; i++) {
    if (labels[i] === name) {
      var x = trace.x && trace.x[i], y = trace.y && trace.y[i], z = trace.z && trace.z[i];
      if (x === null || y === null || z === null ||
          x === undefined || y === undefined || z === undefined) return null;
      x = Number(x); y = Number(y); z = Number(z);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        return {x: x, y: y, z: z};
      }
    }
  }
  return null;
}
function midpointPoint(a, b){
  if (!a && !b) return null;
  if (!a) return b;
  if (!b) return a;
  return {x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2};
}
function subtractPoint(a, b){
  return {x: a.x - b.x, y: a.y - b.y, z: a.z - b.z};
}
function skeletonTransformFromData(data){
  var mode = skeletonSpaceMode();
  var rawPoints = data[1] || {};
  var leftHip = pointByName(rawPoints, 'leftHip');
  var rightHip = pointByName(rawPoints, 'rightHip');
  var leftShoulder = pointByName(rawPoints, 'leftShoulder');
  var rightShoulder = pointByName(rawPoints, 'rightShoulder');
  var nose = pointByName(rawPoints, 'nose');
  var center = midpointPoint(leftHip, rightHip) ||
               midpointPoint(leftShoulder, rightShoulder) ||
               nose || {x: 0, y: 0, z: 0};
  var yaw = 0;
  if (mode === 'body') {
    var left = leftHip || leftShoulder;
    var right = rightHip || rightShoulder;
    if (left && right) {
      var lateral = subtractPoint(left, right);
      if (Math.abs(lateral.x) + Math.abs(lateral.z) > 1e-9) {
        yaw = Math.atan2(lateral.z, lateral.x);
      }
    }
  }
  return {
    mode: mode,
    center: (mode === 'centered' || mode === 'body') ?
      {x: center.x, y: 0, z: center.z} : {x: 0, y: 0, z: 0},
    cos: Math.cos(yaw),
    sin: Math.sin(yaw)
  };
}
function transformArrayTriplet(xValues, yValues, zValues, transform){
  var x = [], y = [], z = [];
  var subjectYaw = skeletonSubjectYawRadians();
  var subjectCos = Math.cos(subjectYaw);
  var subjectSin = Math.sin(subjectYaw);
  for (var i = 0; i < xValues.length; i++) {
    var rawX = xValues[i], rawY = yValues[i], rawZ = zValues[i];
    if (rawX === null || rawY === null || rawZ === null ||
        rawX === undefined || rawY === undefined || rawZ === undefined) {
      x.push(rawX); y.push(rawY); z.push(rawZ);
      continue;
    }
    var dx = Number(rawX) - transform.center.x;
    var dy = Number(rawY) - transform.center.y;
    var dz = Number(rawZ) - transform.center.z;
    if (!Number.isFinite(dx) || !Number.isFinite(dy) || !Number.isFinite(dz)) {
      x.push(rawX); y.push(rawY); z.push(rawZ);
      continue;
    }
    var tx = dx, ty = dy, tz = dz;
    if (transform.mode === 'body') {
      tx = dx * transform.cos + dz * transform.sin;
      tz = -dx * transform.sin + dz * transform.cos;
    }
    x.push(tx * subjectCos + tz * subjectSin);
    y.push(ty);
    z.push(-tx * subjectSin + tz * subjectCos);
  }
  return {x: x, y: y, z: z};
}
function transformedSkeletonFrameData(data){
  var mode = skeletonSpaceMode();
  if (mode === 'world' && Math.abs(skeletonSubjectYawDegrees) < 1e-9) return data;
  var transform = skeletonTransformFromData(data);
  return data.map(function(trace){
    trace = trace || {};
    var transformed = transformArrayTriplet(
      trace.x || [], trace.y || [], trace.z || [], transform);
    return Object.assign({}, trace, transformed);
  });
}
function restyleSkeletonFrame(div, frameIndex, verbose){
  var id = div.dataset.figid;
  var payloads = skeletonFramePayloads[id] || [];
  if (!payloads.length) return false;
  var safeIndex = ((frameIndex % payloads.length) + payloads.length) % payloads.length;
  var data = transformedSkeletonFrameData(payloads[safeIndex] || []);
  var x = [], y = [], z = [], text = [];
  for (var i = 0; i < 6; i++) {
    var trace = data[i] || {};
    x.push(trace.x || []);
    y.push(trace.y || []);
    z.push(trace.z || []);
    text.push(trace.text || []);
  }
  runWithSkeletonViewState(div, function(){
    return Plotly.restyle(div, {x: x, y: y, z: z, text: text}, [0, 1, 2, 3, 4, 5]);
  });
  skeletonPlaybackIndex[id] = safeIndex;
  var frameName = (skeletonFrameNames[id] || [])[safeIndex] || safeIndex;
  var status = document.getElementById('skel-frame-status');
  if (status) status.textContent = id + ' frame ' + frameName;
  updateSkeletonFrameControls(div);
  if (verbose || safeIndex % 15 === 0) {
    var rawPoint = (data[1] && data[1].x && data[1].x.length) ?
      (' raw0=(' + Number(data[1].x[0]).toFixed(4) + ',' +
       Number(data[1].y[0]).toFixed(4) + ',' +
       Number(data[1].z[0]).toFixed(4) + ')') : '';
    skeletonLog(id+' manual frame '+frameName+' index '+safeIndex+'/'+payloads.length+rawPoint);
  }
  return true;
}
function playSkeletonPlots(){
  refreshMounts();
  var plots = Array.prototype.slice.call(
    document.querySelectorAll('.skeleton-variant-block.active .skeleton-plot'));
  skeletonLog('Play clicked; found '+plots.length+' active skeleton plot divs; using manual restyle loop');
  plots.forEach(function(div){
    var id = div.dataset.figid;
    if (!mounted[id]) {
      skeletonLog(id+' is not mounted; skipping');
      return;
    }
    var frames = skeletonFramePayloads[id] || [];
    var step = skeletonFrameStep();
    var delay = skeletonFrameDelay();
    skeletonLog(id+' play request with '+frames.length+' frames, step='+step+', delay='+delay+'ms');
    if (!frames.length) return;
    stopSkeletonTimer(id);
    restyleSkeletonFrame(div, skeletonPlaybackIndex[id] || 0, true);
    skeletonPlaybackTimers[id] = window.setInterval(function(){
      var nextIndex = ((skeletonPlaybackIndex[id] || 0) + step) % frames.length;
      restyleSkeletonFrame(div, nextIndex, false);
    }, delay);
  });
}
function pauseSkeletonPlots(){
  skeletonLog('Pause clicked');
  document.querySelectorAll('.skeleton-plot').forEach(function(div){
    stopSkeletonTimer(div.dataset.figid);
  });
}
function activeSkeletonPlot(){
  var plots = Array.prototype.slice.call(
    document.querySelectorAll('.skeleton-variant-block.active .skeleton-plot'));
  for (var i = 0; i < plots.length; i++) {
    if (mounted[plots[i].dataset.figid]) return plots[i];
  }
  return null;
}
function cloneRange(range){
  return range && range.length === 2 ? [range[0], range[1]] : null;
}
function cloneCamera(camera){
  camera = camera || {};
  var cloned = {
    eye: Object.assign({}, SKELETON_DEFAULT_CAMERA.eye, camera.eye || {}),
    up: Object.assign({}, SKELETON_DEFAULT_CAMERA.up, camera.up || {}),
    projection: Object.assign(
      {}, SKELETON_DEFAULT_CAMERA.projection, camera.projection || {})
  };
  if (camera.center) {
    cloned.center = Object.assign({x: 0, y: 0, z: 0}, camera.center);
  }
  return cloned;
}
function liveScene(div){
  return (div._fullLayout && div._fullLayout.scene) ||
         (div.layout && div.layout.scene) || {};
}
function captureSkeletonViewState(div){
  var scene = liveScene(div);
  return {
    camera: cloneCamera(scene.camera),
    xaxisRange: cloneRange(scene.xaxis && scene.xaxis.range),
    yaxisRange: cloneRange(scene.yaxis && scene.yaxis.range),
    zaxisRange: cloneRange(scene.zaxis && scene.zaxis.range)
  };
}
function rememberSkeletonViewState(div){
  if (!div || !div.dataset) return;
  skeletonViewStateByPlot[div.dataset.figid] = captureSkeletonViewState(div);
}
function restoreSkeletonViewState(div, state){
  if (!div || !state) return;
  var update = {'scene.camera': state.camera};
  if (state.xaxisRange) update['scene.xaxis.range'] = state.xaxisRange;
  if (state.yaxisRange) update['scene.yaxis.range'] = state.yaxisRange;
  if (state.zaxisRange) update['scene.zaxis.range'] = state.zaxisRange;
  return Plotly.relayout(div, update);
}
function runWithSkeletonViewState(div, action){
  var id = div.dataset.figid;
  rememberSkeletonViewState(div);
  var state = skeletonViewStateByPlot[id];
  var result = action();
  Promise.resolve(result).then(function(){
    var latestState = skeletonViewStateByPlot[id] || state;
    return restoreSkeletonViewState(div, latestState);
  }).then(function(){
    rememberSkeletonViewState(div);
  }).catch(function(e){
    console.warn('restore skeleton view state failed', e);
  });
  return result;
}
function vector(x, y, z){ return {x: x, y: y, z: z}; }
function addVec(a, b){ return vector(a.x + b.x, a.y + b.y, a.z + b.z); }
function scaleVec(a, s){ return vector(a.x * s, a.y * s, a.z * s); }
function crossVec(a, b){
  return vector(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
function dotVec(a, b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
function normVec(a){ return Math.sqrt(dotVec(a, a)) || 1; }
function normalizeVec(a){ return scaleVec(a, 1 / normVec(a)); }
function rotateVec(v, axis, angle){
  var u = normalizeVec(axis);
  var c = Math.cos(angle), s = Math.sin(angle);
  return addVec(
    addVec(scaleVec(v, c), scaleVec(crossVec(u, v), s)),
    scaleVec(u, dotVec(u, v) * (1 - c))
  );
}
function currentCamera(div){
  return cloneCamera(liveScene(div).camera);
}
function rotateSkeletonCamera(div, yaw, pitch){
  var cam = currentCamera(div);
  var eye = vector(cam.eye.x, cam.eye.y, cam.eye.z);
  var up = normalizeVec(vector(cam.up.x, cam.up.y, cam.up.z));
  if (yaw) eye = rotateVec(eye, up, yaw);
  if (pitch) {
    var right = normalizeVec(crossVec(eye, up));
    eye = rotateVec(eye, right, pitch);
    up = normalizeVec(rotateVec(up, right, pitch));
  }
  Plotly.relayout(div, {
    'scene.camera': {eye: eye, up: up, projection: cam.projection}
  }).then(function(){
    rememberSkeletonViewState(div);
  });
}
function zoomSkeletonCamera(div, factor){
  var cam = currentCamera(div);
  Plotly.relayout(div, {
    'scene.camera': {
      eye: scaleVec(vector(cam.eye.x, cam.eye.y, cam.eye.z), factor),
      up: cam.up,
      projection: cam.projection
    }
  }).then(function(){
    rememberSkeletonViewState(div);
  });
}
function panSkeletonRanges(div, dxFrac, dyFrac){
  var scene = liveScene(div);
  var xr = scene.xaxis && scene.xaxis.range ? scene.xaxis.range.slice() : null;
  var yr = scene.yaxis && scene.yaxis.range ? scene.yaxis.range.slice() : null;
  if (!xr || !yr) return;
  var dx = (xr[1] - xr[0]) * dxFrac;
  var dy = (yr[1] - yr[0]) * dyFrac;
  Plotly.relayout(div, {
    'scene.xaxis.range': [xr[0] + dx, xr[1] + dx],
    'scene.yaxis.range': [yr[0] + dy, yr[1] + dy]
  }).then(function(){
    rememberSkeletonViewState(div);
  });
}
function resetSkeletonCamera(){
  var div = activeSkeletonPlot();
  if (!div) return;
  Plotly.relayout(div, {'scene.camera': SKELETON_DEFAULT_CAMERA}).then(function(){
    rememberSkeletonViewState(div);
  });
  skeletonLog('camera reset');
}
document.addEventListener('keydown', function(e){
  var target = e.target && e.target.tagName ? e.target.tagName.toLowerCase() : '';
  if (target === 'input' || target === 'select' || target === 'textarea') return;
  var skeletonActive = document.getElementById('view-skeleton').classList.contains('active');
  if (skeletonActive) {
    var key = e.key.toLowerCase();
    var div = activeSkeletonPlot();
    if (!div) return;
    var handled = true;
    if (e.shiftKey && key === 'a') panSkeletonRanges(div, -0.04, 0);
    else if (e.shiftKey && key === 'd') panSkeletonRanges(div, 0.04, 0);
    else if (e.shiftKey && key === 'w') panSkeletonRanges(div, 0, 0.04);
    else if (e.shiftKey && key === 's') panSkeletonRanges(div, 0, -0.04);
    else if (key === 'a') rotateSkeletonCamera(div, -0.08, 0);
    else if (key === 'd') rotateSkeletonCamera(div, 0.08, 0);
    else if (key === 'w') rotateSkeletonCamera(div, 0, -0.06);
    else if (key === 's') rotateSkeletonCamera(div, 0, 0.06);
    else if (key === 'q') zoomSkeletonCamera(div, 0.9);
    else if (key === 'e') zoomSkeletonCamera(div, 1.1);
    else if (key === 'r') resetSkeletonCamera();
    else handled = false;
    if (handled) e.preventDefault();
    return;
  }
  if(document.getElementById('view-group').classList.contains('active')) return;
  if(e.key==='ArrowRight'){stepLandmark(1);e.preventDefault();}
  if(e.key==='ArrowLeft'){stepLandmark(-1);e.preventDefault();}
});
refreshMounts();   // initial mount of the default (Individual) view
</script></body></html>
"""


def write_html(df, summary, pairs, landmarks, out_path,
               iqr_k, min_delta, extras_label, render_group=True,
               fps=30.0, settings=None, cache_db=None,
               simulation_parallel=True, simulation_workers=None,
               simulation_backend="thread",
               progress_callback: Optional[Callable[[str], None]] = None):
    settings = settings or PoseDetectionSettings()
    summary_idx = summary.set_index(["landmark", "variant"], drop=False)
    parts = [HTML_HEAD, _plotlyjs_cdn_tag()]

    pid = 0
    def nid():
        nonlocal pid; pid += 1; return f"p{pid}"

    opts = "".join(f"<option value='{lm}'>{lm}</option>" for lm in landmarks)
    parts.append(
        "<div class='toggle-bar'>"
        "<button id='btn-group'  onclick=\"setView('group')\">Group view</button>"
        "<button id='btn-detail' class='active' onclick=\"setView('detail')\">Individual view</button>"
        "<button id='btn-3d'     onclick=\"setView('3d')\">3D view</button>"
        "<button id='btn-skeleton' onclick=\"setView('skeleton')\">Skeleton view</button>"
        "<span id='nav' class='nav active'>"
        "  <button onclick='stepLandmark(-1)' title='Previous (â†)'>â—€</button>"
        f" <select id='lm-select' onchange='setLandmark(this.value)'>{opts}</select>"
        "  <button onclick='stepLandmark(1)' title='Next (â†’)'>â–¶</button>"
        f" <span id='lm-pos'>1 / {len(landmarks)}</span></span>"
        f"<span class='params'>extras (CSV only): {extras_label} Â· "
        f"outlier: IQRÂ·k={iqr_k:g}, minÎ”={min_delta:g}</span></div>"
    )
    parts.append(_settings_inputs_html(settings))

    # ---- group (lazy, keep-once-mounted) -------------------------------
    _emit_progress(progress_callback, "Building Group view...")
    parts.append("<div id='view-group' class='view'>")
    if render_group:
        parts.append(_lazy_plot_html(build_group_figure(df, pairs), nid(), keep=True))
    else:
        parts.append("<p style='color:#888;padding:20px'>Group view disabled "
                     "(<code>--no-group</code>).</p>")
    parts.append("</div>")

    # ---- individual (lazy, purge on hide) ------------------------------
    _emit_progress(progress_callback, "Building Individual view...")
    parts.append("<div id='view-detail' class='view active'>")
    parts.append(_ranking_table_html(summary))
    for i, lm in enumerate(landmarks):
        active = " active" if i == 0 else ""
        parts.append(f"<div class='lm-block{active}' data-lm='{lm}'>")
        for var in sorted(df.loc[df["landmark"] == lm, "variant"].unique()):
            g = df[(df["landmark"] == lm) & (df["variant"] == var)]
            srow = summary_idx.loc[(lm, var)]
            parts.append("<div class='alpha-block'>")
            parts.append(_quick_glance_html(srow))
            parts.append(_lazy_plot_html(build_detail_figure(g, lm, var), nid()))
            parts.append("<div class='stats-caption'>res = filt âˆ’ raw Â· "
                         "jitterâ†“ = 1 âˆ’ std(Î”filt)/std(Î”raw), avg XYZ Â· "
                         "outliers: |Î”raw| > Q3 + kÂ·IQR and > minÎ”</div>")
            parts.append(_per_axis_table_html(srow))
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div>")

    # ---- 3D (lazy, purge on hide â€” this is what frees WebGL) -----------
    _emit_progress(progress_callback, "Building 3D landmark trajectory views...")
    parts.append("<div id='view-3d' class='view'>")
    for i, lm in enumerate(landmarks):
        active = " active" if i == 0 else ""
        parts.append(f"<div class='lm-block{active}' data-lm='{lm}'>")
        for var in sorted(df.loc[df["landmark"] == lm, "variant"].unique()):
            g = df[(df["landmark"] == lm) & (df["variant"] == var)]
            srow = summary_idx.loc[(lm, var)]
            parts.append("<div class='alpha-block'>")
            parts.append(_quick_glance_html(srow))
            parts.append(_lazy_plot_html(build_3d_figure(g, lm, var), nid()))
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div>")

    # ---- skeleton playback (lazy, purge on hide) -----------------------
    _emit_progress(progress_callback, "Building Skeleton playback view...")
    parts.append("<div id='view-skeleton' class='view'>")
    skeleton_variants = sorted(df["variant"].unique())
    variant_options = "".join(
        f"<option value='{html.escape(var, quote=True)}'>{html.escape(var)}</option>"
        for var in skeleton_variants
    )
    parts.append(
        "<div class='skeleton-controls'>"
        "<button id='skel-play' onclick='playSkeletonPlots()'>Play</button>"
        "<button id='skel-pause' onclick='pauseSkeletonPlots()'>Pause</button>"
        "<label>Variant <select id='skel-variant-select' "
        "onchange='setSkeletonVariant(this.value)'>"
        f"{variant_options}</select></label>"
        "<label><input id='skel-show-raw' type='checkbox' checked "
        "onchange='setSkeletonVisibility()'> Raw</label>"
        "<label><input id='skel-show-filtered' type='checkbox' checked "
        "onchange='setSkeletonVisibility()'> Filtered</label>"
        "<label><input id='skel-show-labels' type='checkbox' "
        "onchange='setSkeletonVisibility()'> Landmark names</label>"
        "<label>Speed <select id='skel-speed' onchange='restartSkeletonPlaybackIfRunning()'>"
        "<option value='0.5'>0.5x</option>"
        "<option value='1' selected>1x</option>"
        "<option value='1.5'>1.5x</option>"
        "<option value='2'>2x</option>"
        "</select></label>"
        "<label>Step <select id='skel-frame-step' onchange='restartSkeletonPlaybackIfRunning()'>"
        "<option value='1' selected>1</option>"
        "<option value='2'>2</option>"
        "<option value='4'>4</option>"
        "<option value='8'>8</option>"
        "</select></label>"
        "<label>Pose frame <select id='skel-space' onchange='setSkeletonSpaceMode()'>"
        "<option value='world' selected>As logged</option>"
        "<option value='centered'>Center hips</option>"
        "<option value='body'>Face lock</option>"
        "</select></label>"
        "<label>Subject yaw "
        "<input id='skel-subject-yaw' type='range' min='-180' max='180' "
        "step='1' value='0' oninput='setSkeletonSubjectYaw(this.value)'>"
        "<span id='skel-subject-yaw-value'>0 deg</span></label>"
        "<button id='skel-twirl' onclick='startSkeletonSubjectSpin()'>Twirl</button>"
        "<button id='skel-stop-twirl' onclick='stopSkeletonSubjectSpin()'>Stop twirl</button>"
        "<button id='skel-reset-camera' onclick='resetSkeletonCamera()' "
        "title='Reset camera; WASD rotate, Shift+WASD pan, Q/E zoom'>Reset camera</button>"
        "<select id='skel-mode' onchange='setSkeletonVisibility()'>"
        "<option value='logged'>Logged playback</option>"
        "<option value='simulation'>Simulation playback</option>"
        "</select>"
        "<span id='skel-frame-status' class='skeleton-status'>Frame -</span>"
        "<button id='skel-clear-log' onclick='clearSkeletonLog()'>Clear log</button>"
        "<span class='skeleton-key raw'>Raw</span>"
        "<span class='skeleton-key logged'>Logged filtered</span>"
        "<span class='skeleton-key sim'>Kalman simulation</span>"
        "</div>"
        "<div class='skeleton-scrubber'>"
        "<span>Elapsed frames</span>"
        "<input id='skel-frame-slider' type='range' min='0' max='0' "
        "step='1' value='0' oninput='seekSkeletonFrame(this.value)' disabled>"
        "<span id='skel-frame-count'>0 / 0</span>"
        "</div>"
        "<div id='skeleton-log' class='skeleton-log'>Skeleton debug log ready. "
        "Open the browser console for the same messages.\n</div>"
    )
    for index, var in enumerate(skeleton_variants):
        g = df[df["variant"] == var]
        active = " active" if index == 0 else ""
        _emit_progress(
            progress_callback,
            f"Building skeleton variant {index + 1}/{len(skeleton_variants)}: {var}",
        )
        parts.append(
            f"<div class='alpha-block skeleton-variant-block{active}' "
            f"data-variant='{html.escape(var, quote=True)}'>"
        )
        parts.append(f"<h3>{html.escape(var)}</h3>")
        parts.append(_lazy_plot_html(
            build_skeleton_figure(
                g,
                var,
                fps=fps,
                settings=settings,
                cache_db=cache_db,
                simulation_parallel=simulation_parallel,
                simulation_workers=simulation_workers,
                simulation_backend=simulation_backend,
                progress_callback=progress_callback,
            ),
            nid(),
            extra_class="skeleton-plot",
        ))
        parts.append("</div>")
    parts.append("</div>")

    tail = HTML_TAIL_TMPL.replace("__LANDMARKS__", json.dumps(landmarks))
    tail = tail.replace("__SKELETON_FRAME_MS__", str(max(int(1000.0 / fps), 1)))
    parts.append(tail)
    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"[info] wrote {out_path}")


def _split_landmarks(value):
    if not value:
        return None
    if isinstance(value, str):
        parts = re.split(r"[,\s]+", value.strip())
    else:
        parts = value
    landmarks = [str(part).strip() for part in parts if str(part).strip()]
    return landmarks or None


def default_output_paths_for_log(log_path):
    log_path = Path(log_path)
    html_path = log_path.with_name(f"{log_path.stem}_world_filter_review.html")
    stem = html_path.with_suffix("")
    return html_path, Path(f"{stem}_frames.csv"), Path(f"{stem}_summary.csv")


def generate_report(
    logfile,
    landmarks=None,
    out=Path("world_filter_review.html"),
    csv_frames=None,
    csv_summary=None,
    tag_re=r"WORLD_\w+",
    schema=None,
    iqr_k=1.5,
    min_delta=0.0,
    render_group=True,
    fps=30.0,
    settings=None,
    cache_db=None,
    simulation_parallel=True,
    simulation_workers=None,
    simulation_backend="thread",
    progress_callback: Optional[Callable[[str], None]] = None,
    open_report=False,
):
    logfile = Path(logfile)
    out = Path(out)
    csv_frames = Path(csv_frames) if csv_frames else None
    csv_summary = Path(csv_summary) if csv_summary else None
    cache_db = Path(cache_db) if cache_db else default_cache_db_for_report(out)
    settings = settings or PoseDetectionSettings()
    extra_names = [s.strip() for s in schema.split(",")] if schema else []
    _emit_progress(progress_callback, f"Parsing log file: {logfile}")
    df = parse_log(logfile, tag_re=tag_re, extra_names=extra_names)
    _emit_progress(progress_callback, f"Parsed {len(df)} WORLD rows")

    all_lms = sorted(df["landmark"].unique())
    lms = _split_landmarks(landmarks)
    if lms:
        selected_lms = [lm for lm in lms if lm in all_lms]
        missing = set(lms) - set(selected_lms)
        if missing:
            print(f"[warn] not in log: {sorted(missing)}", file=sys.stderr)
        df = df[df["landmark"].isin(selected_lms)].reset_index(drop=True)
        lms = selected_lms
    else:
        lms = all_lms

    if not lms:
        raise ValueError("No selected landmarks were found in the log")

    _emit_progress(progress_callback, "Computing residual, jitter, and outlier statistics...")
    df = enrich(df, iqr_k=iqr_k, min_delta=min_delta)
    _emit_progress(progress_callback, "Summarizing landmarks and variants...")
    summary = summarize(df)
    pairs = sorted(set(zip(df["landmark"], df["variant"])))

    core = {"tag", "frameNo", "landmark", "variant", "alpha",
            "rawX", "rawY", "rawZ", "filtX", "filtY", "filtZ"}
    extras_found = sorted(c for c in df.columns
                          if c not in core and not c.startswith(
                              ("res", "absRes", "dRaw", "dFilt", "outlier")))
    extras_label = ", ".join(extras_found) if extras_found else "none"

    print(f"[info] {len(df)} rows, {len(lms)} landmarks, "
          f"{len(pairs)} (landmark,variant) pairs, "
          f"frames {df['frameNo'].min()}..{df['frameNo'].max()}")
    print(f"[info] extra columns captured (CSV only): {extras_label}")

    stem = out.with_suffix("")
    frames_csv = csv_frames or Path(f"{stem}_frames.csv")
    summary_csv = csv_summary or Path(f"{stem}_summary.csv")
    _emit_progress(progress_callback, f"Writing frames CSV: {frames_csv}")
    df.to_csv(frames_csv, index=False)
    _emit_progress(progress_callback, f"Writing summary CSV: {summary_csv}")
    summary.to_csv(summary_csv, index=False)
    print(f"[info] wrote {frames_csv}")
    print(f"[info] wrote {summary_csv}")
    print(f"[info] simulation cache DB: {cache_db}")

    _emit_progress(progress_callback, f"Building HTML report: {out}")
    write_html(df, summary, pairs, lms, out,
               iqr_k=iqr_k, min_delta=min_delta,
               extras_label=extras_label,
               render_group=render_group,
               fps=fps,
               settings=settings,
               cache_db=cache_db,
               simulation_parallel=simulation_parallel,
               simulation_workers=simulation_workers,
               simulation_backend=simulation_backend,
               progress_callback=progress_callback)
    _emit_progress(progress_callback, f"Finished report: {out}")

    if open_report:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())

    return {
        "frames": df,
        "summary": summary,
        "pairs": pairs,
        "landmarks": lms,
        "html": out,
        "frames_csv": frames_csv,
        "summary_csv": summary_csv,
        "cache_db": cache_db,
        "extras_label": extras_label,
    }


def launch_gui():
    import queue
    import threading
    import traceback
    import webbrowser
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("WORLD Filter Visualizer")
    root.geometry("820x760")

    default_settings = PoseDetectionSettings()
    log_var = tk.StringVar()
    out_var = tk.StringVar(value=str(Path("world_filter_review.html")))
    frames_var = tk.StringVar()
    summary_var = tk.StringVar()
    cache_var = tk.StringVar(value=str(default_cache_db_for_report(Path("world_filter_review.html"))))
    landmarks_var = tk.StringVar()
    tag_var = tk.StringVar(value=r"WORLD_\w+")
    schema_var = tk.StringVar(value="vel")
    iqr_var = tk.StringVar(value="1.5")
    min_delta_var = tk.StringVar(value="0.0")
    fps_var = tk.StringVar(value="30.0")
    simulation_backend_var = tk.StringVar(value="thread")
    simulation_workers_var = tk.StringVar(value="4")
    render_group_var = tk.BooleanVar(value=True)
    open_var = tk.BooleanVar(value=True)
    status_var = tk.StringVar(value="Choose a log file, then generate the report.")
    available_presets: Dict[str, PoseDetectionSettings] = {"Defaults": default_settings}
    preset_var = tk.StringVar(value="Defaults")
    active_preset_var = tk.StringVar(value="Coefficient set: Defaults")
    preset_state = {"name": "Defaults", "updating": False}

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)

    def choose_log():
        path = filedialog.askopenfilename(
            title="Select WORLD_* log file",
            filetypes=[("Log/Text files", "*.log *.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        log_var.set(path)
        html_path, frames_path, summary_path = default_output_paths_for_log(path)
        out_var.set(str(html_path))
        frames_var.set(str(frames_path))
        summary_var.set(str(summary_path))
        cache_var.set(str(default_cache_db_for_report(html_path)))

    def choose_save(var, title, default_ext, filetypes):
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=default_ext,
            filetypes=filetypes,
        )
        if path:
            var.set(path)

    def choose_html():
        choose_save(out_var, "Save HTML report", ".html",
                    [("HTML files", "*.html"), ("All files", "*.*")])
        if out_var.get().strip():
            stem = Path(out_var.get().strip()).with_suffix("")
            frames_var.set(str(Path(f"{stem}_frames.csv")))
            summary_var.set(str(Path(f"{stem}_summary.csv")))
            cache_var.set(str(default_cache_db_for_report(Path(out_var.get().strip()))))

    def add_file_row(parent, row, label, var, browse_command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(parent, text="Browse", command=browse_command).grid(row=row, column=2, pady=3)

    files = ttk.LabelFrame(main, text="Files", padding=8)
    files.pack(fill="x")
    files.columnconfigure(1, weight=1)
    add_file_row(files, 0, "Log file", log_var, choose_log)
    add_file_row(
        files, 1, "HTML output", out_var, choose_html,
    )
    add_file_row(
        files, 2, "Frames CSV", frames_var,
        lambda: choose_save(frames_var, "Save frames CSV", ".csv",
                            [("CSV files", "*.csv"), ("All files", "*.*")]),
    )
    add_file_row(
        files, 3, "Summary CSV", summary_var,
        lambda: choose_save(summary_var, "Save summary CSV", ".csv",
                            [("CSV files", "*.csv"), ("All files", "*.*")]),
    )
    add_file_row(
        files, 4, "Cache DB", cache_var,
        lambda: choose_save(cache_var, "Save simulation cache DB", ".sqlite",
                            [("SQLite files", "*.sqlite"), ("All files", "*.*")]),
    )

    options = ttk.LabelFrame(main, text="Report options", padding=8)
    options.pack(fill="x", pady=(10, 0))
    for col in (1, 3):
        options.columnconfigure(col, weight=1)
    option_rows = [
        ("Landmarks", landmarks_var, "blank = all"),
        ("Tag regex", tag_var, ""),
        ("Extra schema", schema_var, "example: vel"),
        ("IQR k", iqr_var, ""),
        ("Min delta", min_delta_var, ""),
        ("FPS", fps_var, ""),
    ]
    for idx, (label, var, hint) in enumerate(option_rows):
        row, side = divmod(idx, 2)
        col = side * 2
        ttk.Label(options, text=label).grid(row=row, column=col, sticky="w", pady=3)
        entry = ttk.Entry(options, textvariable=var)
        entry.grid(row=row, column=col + 1, sticky="ew", padx=6, pady=3)
    ttk.Checkbutton(options, text="Build Group view", variable=render_group_var).grid(
        row=3, column=0, sticky="w", pady=4)
    ttk.Checkbutton(options, text="Open report when done", variable=open_var).grid(
        row=3, column=1, sticky="w", pady=4)
    ttk.Label(options, text="Sim backend").grid(row=4, column=0, sticky="w", pady=3)
    ttk.Combobox(
        options,
        textvariable=simulation_backend_var,
        values=["thread", "process"],
        state="readonly",
        width=12,
    ).grid(row=4, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(options, text="Sim workers").grid(row=4, column=2, sticky="w", pady=3)
    ttk.Entry(options, textvariable=simulation_workers_var, width=10).grid(
        row=4, column=3, sticky="w", padx=6, pady=3)

    coef_box = ttk.LabelFrame(main, text="Filter coefficients", padding=8)
    coef_box.pack(fill="both", expand=True, pady=(10, 0))
    canvas = tk.Canvas(coef_box, highlightthickness=0, height=300)
    scrollbar = ttk.Scrollbar(coef_box, orient="vertical", command=canvas.yview)
    coef_frame = ttk.Frame(canvas)
    coef_frame.bind(
        "<Configure>",
        lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=coef_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def mark_coefficients_custom(*_args):
        if preset_state["updating"]:
            return
        name = preset_state["name"]
        if not str(name).startswith("Custom"):
            name = f"Custom from {name}"
            preset_state["name"] = name
            active_preset_var.set(f"Coefficient set: {name}")
        else:
            active_preset_var.set(f"Coefficient set: {name}")

    coef_vars = {}
    for idx, f in enumerate(fields(PoseDetectionSettings)):
        row, side = divmod(idx, 2)
        col = side * 2
        var = tk.StringVar(value=str(getattr(default_settings, f.name)))
        var.trace_add("write", mark_coefficients_custom)
        coef_vars[f.name] = var
        ttk.Label(coef_frame, text=f.name).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(coef_frame, textvariable=var, width=14).grid(row=row, column=col + 1, sticky="ew", padx=(0, 12), pady=2)

    def apply_settings_to_fields(settings, preset_name="Custom"):
        values = asdict(settings)
        preset_state["updating"] = True
        try:
            for name, var in coef_vars.items():
                var.set(str(values[name]))
        finally:
            preset_state["updating"] = False
        preset_state["name"] = preset_name
        active_preset_var.set(f"Coefficient set: {preset_name}")
        if preset_name in available_presets:
            preset_var.set(preset_name)

    def refresh_preset_combo():
        names = sorted(available_presets)
        preset_combo.configure(values=names)
        if preset_var.get() not in names and names:
            preset_var.set(names[0])

    def apply_selected_preset():
        name = preset_var.get()
        settings = available_presets.get(name)
        if settings is None:
            messagebox.showerror("Filter coefficients", f"Unknown preset: {name}")
            return
        apply_settings_to_fields(settings, name)
        status_var.set(
            f"Loaded coefficients: {name}. Edit any field to create a custom run."
        )

    def choose_preset_name(presets):
        names = sorted(presets)
        if len(names) == 1:
            return names[0]
        dialog = tk.Toplevel(root)
        dialog.title("Choose coefficient preset")
        dialog.transient(root)
        dialog.grab_set()
        dialog.resizable(False, False)
        selected = tk.StringVar(value=names[0])
        ttk.Label(
            dialog,
            text="Choose the preset to load into the coefficient fields:",
            padding=(12, 12, 12, 4),
        ).pack(fill="x")
        combo = ttk.Combobox(dialog, textvariable=selected, values=names, state="readonly", width=36)
        combo.pack(fill="x", padx=12, pady=(0, 12))
        result = {"name": None}

        def accept():
            result["name"] = selected.get()
            dialog.destroy()

        def cancel():
            dialog.destroy()

        buttons = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Load", command=accept).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=cancel).pack(side="left", padx=8)
        combo.focus_set()
        dialog.bind("<Return>", lambda _event: accept())
        dialog.bind("<Escape>", lambda _event: cancel())
        root.wait_window(dialog)
        return result["name"]

    def import_coefficients():
        path = filedialog.askopenfilename(
            title="Import filter coefficient preset",
            filetypes=[
                ("Settings documents", "*.json *.csv *.kt *.kts *.txt"),
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("Kotlin/Text files", "*.kt *.kts *.txt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            presets = parse_pose_detection_settings_document(Path(path))
            if not presets:
                raise ValueError("No PoseDetectionSettings values were found")
            selected = choose_preset_name(presets)
            if not selected:
                return
            available_presets.update(presets)
            refresh_preset_combo()
            preset_var.set(selected)
            apply_settings_to_fields(available_presets[selected], selected)
            status_var.set(
                f"Imported {len(presets)} preset(s); loaded coefficients: {selected}"
            )
        except Exception as exc:
            messagebox.showerror("Import coefficients", str(exc))

    coef_actions = ttk.Frame(main)
    coef_actions.pack(fill="x", pady=(8, 0))
    ttk.Label(coef_actions, text="Preset").pack(side="left")
    preset_combo = ttk.Combobox(
        coef_actions,
        textvariable=preset_var,
        values=sorted(available_presets),
        state="readonly",
        width=26,
    )
    preset_combo.pack(side="left", padx=(6, 4))
    preset_combo.bind("<<ComboboxSelected>>", lambda _event: apply_selected_preset())
    ttk.Button(
        coef_actions,
        text="Apply preset",
        command=apply_selected_preset,
    ).pack(side="left", padx=(0, 8))
    ttk.Button(
        coef_actions,
        text="Import coefficients...",
        command=import_coefficients,
    ).pack(side="left")
    ttk.Button(
        coef_actions,
        text="Reset coefficients",
        command=lambda: apply_settings_to_fields(available_presets["Defaults"], "Defaults"),
    ).pack(side="left", padx=8)
    ttk.Label(coef_actions, textvariable=active_preset_var).pack(side="left", padx=8)

    def run_generation():
        try:
            if not log_var.get().strip():
                raise ValueError("Select a log file first")
            settings = parse_pose_detection_settings(
                {name: var.get() for name, var in coef_vars.items()}
            )
            settings_label = preset_state["name"]
            settings_hash = _settings_signature(settings)[:8]
        except Exception as exc:
            messagebox.showerror("WORLD Filter Visualizer", str(exc))
            return

        progress_dialog = tk.Toplevel(root)
        progress_dialog.title("Generating WORLD Filter Report")
        progress_dialog.transient(root)
        progress_dialog.geometry("720x360")
        progress_dialog.protocol("WM_DELETE_WINDOW", lambda: None)

        progress_status_var = tk.StringVar(value="Starting report generation...")
        ttk.Label(
            progress_dialog,
            textvariable=progress_status_var,
            padding=(12, 12, 12, 4),
        ).pack(fill="x")
        spinner = ttk.Progressbar(progress_dialog, mode="indeterminate")
        spinner.pack(fill="x", padx=12, pady=(0, 8))
        spinner.start(12)
        log_text = tk.Text(progress_dialog, height=12, wrap="word")
        log_text.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        log_text.configure(state="disabled")
        close_button = ttk.Button(progress_dialog, text="Close", state="disabled")
        close_button.pack(anchor="e", padx=12, pady=(0, 12))

        messages = queue.Queue()

        def add_log_line(message):
            log_text.configure(state="normal")
            log_text.insert("end", f"{message}\n")
            log_text.see("end")
            log_text.configure(state="disabled")

        def progress(message):
            messages.put(("progress", message))

        def close_progress():
            progress_dialog.destroy()

        close_button.configure(command=close_progress)
        status_var.set(f"Generating report with {settings_label} ({settings_hash})...")
        worker_state = {"complete": False}

        def worker():
            try:
                progress("Worker started")
                progress(f"Using coefficient set: {settings_label} ({settings_hash})")
                result = generate_report(
                    logfile=Path(log_var.get().strip()),
                    landmarks=landmarks_var.get(),
                    out=Path(out_var.get().strip() or "world_filter_review.html"),
                    csv_frames=Path(frames_var.get().strip()) if frames_var.get().strip() else None,
                    csv_summary=Path(summary_var.get().strip()) if summary_var.get().strip() else None,
                    tag_re=tag_var.get().strip() or r"WORLD_\w+",
                    schema=schema_var.get().strip() or None,
                    iqr_k=float(iqr_var.get()),
                    min_delta=float(min_delta_var.get()),
                    render_group=render_group_var.get(),
                    fps=float(fps_var.get()),
                    settings=settings,
                    cache_db=Path(cache_var.get().strip()) if cache_var.get().strip() else None,
                    simulation_parallel=True,
                    simulation_workers=(
                        int(simulation_workers_var.get())
                        if simulation_workers_var.get().strip()
                        else None
                    ),
                    simulation_backend=simulation_backend_var.get(),
                    progress_callback=progress,
                    open_report=False,
                )
                messages.put(("done", {
                    "html": str(result["html"]),
                    "frames_csv": str(result["frames_csv"]),
                    "summary_csv": str(result["summary_csv"]),
                    "cache_db": str(result["cache_db"]),
                }))
            except Exception as exc:
                messages.put(("error", (exc, traceback.format_exc())))

        worker_thread = threading.Thread(target=worker, daemon=True)

        def finish_progress_dialog(message):
            worker_state["complete"] = True
            spinner.stop()
            progress_status_var.set(message)
            close_button.configure(state="normal")
            progress_dialog.protocol("WM_DELETE_WINDOW", close_progress)

        def poll_messages():
            try:
                while True:
                    kind, payload = messages.get_nowait()
                    if kind == "progress":
                        progress_status_var.set(str(payload))
                        add_log_line(str(payload))
                    elif kind == "done":
                        html_path = Path(payload["html"])
                        finish_progress_dialog(f"Done: {html_path}")
                        add_log_line(f"Done: {html_path}")
                        add_log_line(f"Frames CSV: {payload['frames_csv']}")
                        add_log_line(f"Summary CSV: {payload['summary_csv']}")
                        add_log_line(f"Cache DB: {payload['cache_db']}")
                        status_var.set(f"Wrote {html_path}")
                        if open_var.get():
                            threading.Thread(
                                target=lambda: webbrowser.open(html_path.resolve().as_uri()),
                                daemon=True,
                            ).start()
                    elif kind == "error":
                        exc, tb = payload
                        finish_progress_dialog("Generation failed")
                        add_log_line(tb)
                        status_var.set("Generation failed")
                        messagebox.showerror("WORLD Filter Visualizer", str(exc))
            except queue.Empty:
                pass

            if (
                not worker_state["complete"] and
                not worker_thread.is_alive() and
                messages.empty()
            ):
                finish_progress_dialog("Generation worker exited without a completion message")
                add_log_line("Generation worker exited without a completion message.")
                status_var.set("Generation status unknown")

            if progress_dialog.winfo_exists() and not worker_state["complete"]:
                root.after(100, poll_messages)

        worker_thread.start()
        root.after(100, poll_messages)

    actions = ttk.Frame(main)
    actions.pack(fill="x", pady=(10, 0))
    ttk.Button(actions, text="Generate report", command=run_generation).pack(side="left")
    ttk.Label(actions, textvariable=status_var).pack(side="left", padx=12)

    root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Schema-flexible WORLD_* filter review (lazy-rendered 2D + 3D).")
    ap.add_argument("logfile", type=Path, nargs="?")
    ap.add_argument("--gui", action="store_true",
                    help="Open the file/options GUI instead of running headless.")
    ap.add_argument("--landmarks", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("world_filter_review.html"))
    ap.add_argument("--csv-frames", type=Path, default=None)
    ap.add_argument("--csv-summary", type=Path, default=None)
    ap.add_argument("--tag-re", default=r"WORLD_\w+")
    ap.add_argument("--schema", default=None,
                    help="Names for extra groups between filt and alpha, e.g. 'vel'.")
    ap.add_argument("--iqr-k", type=float, default=1.5)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Playback/simulation frame rate used for skeleton dt.")
    ap.add_argument("--cache-db", type=Path, default=None,
                    help="SQLite cache for simulated Kalman outputs.")
    ap.add_argument("--simulation-backend", choices=["thread", "process"],
                    default="thread",
                    help="Parallel backend for per-landmark simulation.")
    ap.add_argument("--simulation-workers", type=int, default=4,
                    help="Number of parallel simulation workers.")
    ap.add_argument("--no-simulation-parallel", action="store_true",
                    help="Run simulation sequentially.")
    ap.add_argument("--no-group", action="store_true",
                    help="Skip building the Group grid (fastest load, smallest HTML).")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    if args.gui or args.logfile is None:
        launch_gui()
        return

    generate_report(
        logfile=args.logfile,
        landmarks=args.landmarks,
        out=args.out,
        csv_frames=args.csv_frames,
        csv_summary=args.csv_summary,
        tag_re=args.tag_re,
        schema=args.schema,
        iqr_k=args.iqr_k,
        min_delta=args.min_delta,
        render_group=not args.no_group,
        fps=args.fps,
        settings=PoseDetectionSettings(),
        cache_db=args.cache_db,
        simulation_parallel=not args.no_simulation_parallel,
        simulation_workers=args.simulation_workers,
        simulation_backend=args.simulation_backend,
        open_report=args.open,
    )


if __name__ == "__main__":
    main()


