#!/usr/bin/env python3
"""
generate_prab_mode_mixing_appendices_v8.py

PRAB appendix generator, Overleaf test-image mode.

Purpose
-------
Creates one appendix entry per stored crossing data row/record.

For this test, every entry uses the same four PNG filenames:
    iris_1.png
    iris_2.png
    longitudinal_mid.png
    transverse_mid.png

So only four test PNGs need to be uploaded to Overleaf.

Homotypic rows
--------------
- monopole--monopole: loss row only
- dipole--dipole: kick row only
- quadrupole--quadrupole: Kxx, Kyy, Kxy rows only

Heterotypic rows
----------------
- loss, kick, Kxx, Kyy, Kxy rows

v8 is deliberately flexible about column names. It tries:
1. metric-specific keys, e.g. loss_1, k_parallel_E1, Kxx_plus
2. generic homotypic keys, e.g. parent_1, parent_2, plus, minus, R_max
3. Rmax computation from E1/E2/E+/E- if no stored R is found

Edit ROOT_DIR and OUT_TEX inside main(), then run in PyCharm.
"""

from __future__ import annotations

import csv
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TEST_PLOT_FILENAMES = {
    "iris_1": "iris_1.png",
    "iris_2": "iris_2.png",
    "longitudinal_mid": "longitudinal_mid.png",
    "transverse_mid": "transverse_mid.png",
}


APPENDIX_CLASSES = {
    "mono_mono": {
        "family": "homotypic",
        "title": "TM Monopole--TM Monopole",
        "display": "monopole--monopole",
        "csv_rel": "homotypic_monopoles/postprocess/monopole_enhancement_summary.csv",
        "rows": ["loss"],
    },
    "dipole_dipole": {
        "family": "homotypic",
        "title": "TM Dipole--TM Dipole",
        "display": "dipole--dipole",
        "csv_rel": "homotypic_dipoles/postprocess/dipole_enhancement_summary.csv",
        "rows": ["kick"],
    },
    "quad_quad": {
        "family": "homotypic",
        "title": "TM Quadrupole--TM Quadrupole",
        "display": "quadrupole--quadrupole",
        "csv_rel": "homotypic_quadrupoles/postprocess/quadrupole_enhancement_summary.csv",
        "rows": ["Kxx", "Kyy", "Kxy"],
    },
    "mono_dipole": {
        "family": "heterotypic",
        "title": "TM Monopole--TM Dipole",
        "display": "monopole--dipole",
        "rows": ["loss", "kick", "Kxx", "Kyy", "Kxy"],
    },
    "mono_quad": {
        "family": "heterotypic",
        "title": "TM Monopole--TM Quadrupole",
        "display": "monopole--quadrupole",
        "rows": ["loss", "kick", "Kxx", "Kyy", "Kxy"],
    },
    "dipole_quad": {
        "family": "heterotypic",
        "title": "TM Dipole--TM Quadrupole",
        "display": "dipole--quadrupole",
        "rows": ["loss", "kick", "Kxx", "Kyy", "Kxy"],
    },
}


HETEROTYPIC_PKL_REL = "heterotypic_crossings/all_heterotypic_multipole_analyses.pkl"


KEY_ALIASES = {
    "mode_1": [
        "mode_1", "mode1", "mode_i", "E1_mode", "parent_1_mode", "parent1_mode",
        "mode_label_1", "mode_label_i", "label_1", "mnp_1", "mnp_i",
        "mode_i_label", "mode_a", "E1_label", "E_1_label",
    ],
    "mode_2": [
        "mode_2", "mode2", "mode_j", "E2_mode", "parent_2_mode", "parent2_mode",
        "mode_label_2", "mode_label_j", "label_2", "mnp_2", "mnp_j",
        "mode_j_label", "mode_b", "E2_label", "E_2_label",
    ],
    "ell": [
        "ell", "l", "length_factor", "L_factor", "l_factor",
        "crossing_length_factor", "length", "best_ell",
    ],
    "f_hat": [
        "f_hat", "freq_hat", "frequency_hat", "f_norm",
        "normalised_frequency", "normalized_frequency", "best_f_hat",
    ],
}


METRIC_SPECS = {
    "loss": {
        "label": r"$k_{\parallel}^{(1)}$",
        "units": r"$\mathrm{V/pC/m}$",
        "patterns": ["loss", "k_parallel", "kpar", "kpara", "k_par", "parallel"],
        "r_patterns": ["r_loss", "r_k_parallel", "r_kpar", "loss_rmax", "rmax_loss"],
    },
    "kick": {
        "label": r"$k_{\perp}^{(2)}$",
        "units": r"$\mathrm{V/pC/m^2}$",
        "patterns": ["kick", "k_perp", "kperp", "k_perpendicular", "perp"],
        "r_patterns": ["r_kick", "r_k_perp", "r_kperp", "kick_rmax", "rmax_kick"],
    },
    "Kxx": {
        "label": r"$K_{xx}^{(3)}$",
        "units": r"$\mathrm{V/pC/m^3}$",
        "patterns": ["kxx", "Kxx"],
        "r_patterns": ["r_kxx", "r_Kxx", "kxx_rmax", "rmax_kxx"],
    },
    "Kyy": {
        "label": r"$K_{yy}^{(3)}$",
        "units": r"$\mathrm{V/pC/m^3}$",
        "patterns": ["kyy", "Kyy"],
        "r_patterns": ["r_kyy", "r_Kyy", "kyy_rmax", "rmax_kyy"],
    },
    "Kxy": {
        "label": r"$K_{xy}^{(3)}$",
        "units": r"$\mathrm{V/pC/m^3}$",
        "patterns": ["kxy", "Kxy"],
        "r_patterns": ["r_kxy", "r_Kxy", "kxy_rmax", "rmax_kxy"],
    },
}


STATE_PATTERNS = {
    "E1": ["e1", "e_1", "parent1", "parent_1", "mode1", "mode_1", "field1", "field_1", "1"],
    "E2": ["e2", "e_2", "parent2", "parent_2", "mode2", "mode_2", "field2", "field_2", "2"],
    "Eplus": ["eplus", "e_plus", "plus", "mixed_plus", "mix_plus", "sum"],
    "Eminus": ["eminus", "e_minus", "minus", "mixed_minus", "mix_minus", "diff"],
}


GENERIC_STATE_COLUMNS = {
    "E1": [
        "metric_1", "value_1", "parent_1_value", "parent1_value",
        "parent_1_metric", "parent1_metric", "q_1", "q1",
        "e1_value", "e_1_value", "e1_metric", "e_1_metric",
        "parent_1", "parent1",
    ],
    "E2": [
        "metric_2", "value_2", "parent_2_value", "parent2_value",
        "parent_2_metric", "parent2_metric", "q_2", "q2",
        "e2_value", "e_2_value", "e2_metric", "e_2_metric",
        "parent_2", "parent2",
    ],
    "Eplus": [
        "metric_plus", "value_plus", "plus_value", "plus_metric",
        "mixed_plus", "mix_plus", "q_plus", "qplus",
        "eplus_value", "e_plus_value", "eplus_metric", "e_plus_metric",
        "plus", "eplus", "e_plus",
    ],
    "Eminus": [
        "metric_minus", "value_minus", "minus_value", "minus_metric",
        "mixed_minus", "mix_minus", "q_minus", "qminus",
        "eminus_value", "e_minus_value", "eminus_metric", "e_minus_metric",
        "minus", "eminus", "e_minus",
    ],
}


@dataclass
class CrossingEntry:
    family: str
    class_key: str
    values: dict[str, Any] = field(default_factory=dict)


def normal_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def is_missing_or_array_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in ("", "--", "nan", "NaN", "None")
    shape = getattr(value, "shape", None)
    if shape is not None:
        return shape != ()
    if isinstance(value, (list, tuple, dict, set)):
        return True
    return False


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(obj, dict):
        return out

    for k, v in obj.items():
        k = str(k)
        full = f"{prefix}.{k}" if prefix else k

        if isinstance(v, dict):
            out.update(flatten(v, full))
        elif isinstance(v, (list, tuple)) and all(isinstance(x, dict) for x in v):
            for i, item in enumerate(v):
                out.update(flatten(item, f"{full}.{i}"))
        else:
            out[full] = v
            out[k] = v
            out[normal_key(full)] = v
            out[normal_key(k)] = v

    return out


def get_value(values: dict[str, Any], canonical: str) -> Any:
    for key in KEY_ALIASES.get(canonical, [canonical]):
        for candidate in (key, normal_key(key)):
            if candidate in values and not is_missing_or_array_like(values[candidate]):
                return values[candidate]

    lower = {str(k).lower(): v for k, v in values.items()}
    for key in KEY_ALIASES.get(canonical, [canonical]):
        lk = str(key).lower()
        if lk in lower and not is_missing_or_array_like(lower[lk]):
            return lower[lk]

    return None


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[warning] Missing CSV: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = [flatten(dict(row)) for row in csv.DictReader(f)]

    if rows:
        print("  sample columns:", list(rows[0].keys())[:40])
    return rows


def load_pickle(path: Path) -> Any:
    if not path.exists():
        print(f"[warning] Missing PKL: {path}")
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            s = value.strip().replace(",", "")
            s = s.replace("×10^", "e")
            s = s.replace(r"\times10^{", "e").replace("}", "")
            if s in ("", "--", "nan", "NaN", "None"):
                return None
            return float(s)
        return float(value)
    except Exception:
        return None


def fmt_num(value: Any, sig: int = 3) -> str:
    x = to_float(value)
    if x is None or not math.isfinite(x):
        return "--"
    if x == 0:
        return "0"

    ax = abs(x)
    if 1e-2 <= ax < 1e3:
        return f"{x:.4g}"

    exponent = int(math.floor(math.log10(ax)))
    mantissa = x / (10 ** exponent)
    return rf"{mantissa:.{sig}g}\times10^{{{exponent}}}"


def fmt_ratio(value: Any) -> str:
    x = to_float(value)
    if x is None or not math.isfinite(x):
        return "--"
    return f"{x:.3g}"


def compact_mode_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)

    m = re.search(r"TM[_\s-]*(\d{3})", text, flags=re.IGNORECASE)
    if m:
        return f"TM{m.group(1)}"

    m = re.fullmatch(r"\s*(\d{3})\s*", text)
    if m:
        return f"TM{m.group(1)}"

    return text.strip() or None


def latex_mode_label(value: Any) -> str:
    label = compact_mode_label(value)
    if not label:
        return "--"

    m = re.search(r"TM(\d{3})", label, flags=re.IGNORECASE)
    if m:
        return rf"$TM_{{{m.group(1)}}}$"

    return latex_escape(label)


def latex_escape(text: Any) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def metric_score(key: str, metric: str) -> int:
    nk = normal_key(key)
    score = 0
    for pat in METRIC_SPECS[metric]["patterns"]:
        npat = normal_key(pat)
        if nk == npat:
            score += 30
        elif npat in nk:
            score += 15
    return score


def state_score(key: str, state: str) -> int:
    nk = normal_key(key)
    score = 0
    for pat in STATE_PATTERNS[state]:
        npat = normal_key(pat)
        if nk == npat:
            score += 20
        elif nk.endswith("_" + npat) or nk.startswith(npat + "_"):
            score += 12
        elif npat in nk:
            score += 5

    if state == "E1" and re.search(r"(^|_)e?1($|_)", nk):
        score += 10
    if state == "E2" and re.search(r"(^|_)e?2($|_)", nk):
        score += 10
    if state == "Eplus" and ("plus" in nk or "e_plus" in nk):
        score += 10
    if state == "Eminus" and ("minus" in nk or "e_minus" in nk):
        score += 10

    return score


def find_metric_value(values: dict[str, Any], metric: str, state: str) -> Any:
    best_key = None
    best_score = -1

    for key, value in values.items():
        if is_missing_or_array_like(value) or to_float(value) is None:
            continue

        mscore = metric_score(key, metric)
        sscore = state_score(key, state)

        if mscore <= 0 or sscore <= 0:
            continue

        total = mscore + sscore
        if "." not in str(key):
            total += 2

        # Avoid mode label numeric-ish false positives.
        nk = normal_key(key)
        if any(tok in nk for tok in ["mode", "label", "mnp"]):
            total -= 50

        if total > best_score:
            best_score = total
            best_key = key

    return values[best_key] if best_key is not None else None


def find_generic_state_value(values: dict[str, Any], state: str) -> Any:
    aliases = GENERIC_STATE_COLUMNS[state]

    # Exact alias match.
    for alias in aliases:
        na = normal_key(alias)
        for key, value in values.items():
            if is_missing_or_array_like(value) or to_float(value) is None:
                continue
            if normal_key(key) == na:
                return value

    # Soft alias match.
    best_key = None
    best_score = -1
    for key, value in values.items():
        if is_missing_or_array_like(value) or to_float(value) is None:
            continue

        nk = normal_key(key)
        score = 0
        for alias in aliases:
            na = normal_key(alias)
            if nk == na:
                score += 30
            elif nk.endswith("_" + na) or nk.startswith(na + "_"):
                score += 15
            elif na in nk:
                score += 6

        if any(tok in nk for tok in ["rmax", "ratio", "enhancement", "mode", "label", "mnp"]):
            score -= 25

        if score > best_score:
            best_score = score
            best_key = key

    return values[best_key] if best_key is not None and best_score > 0 else None


def find_ratio_value(values: dict[str, Any], metric: str | None = None) -> Any:
    aliases = ["R_max", "Rmax", "ratio_max", "max_ratio", "enhancement", "max_enhancement", "mixed_max", "R"]

    if metric is not None:
        aliases = METRIC_SPECS[metric]["r_patterns"] + aliases

    for alias in aliases:
        na = normal_key(alias)
        for key, value in values.items():
            if is_missing_or_array_like(value) or to_float(value) is None:
                continue
            if normal_key(key) == na:
                return value

    best_key = None
    best_score = -1
    for key, value in values.items():
        if is_missing_or_array_like(value) or to_float(value) is None:
            continue

        nk = normal_key(key)
        score = 0
        if metric is not None and metric_score(key, metric) > 0:
            score += 10
        if "rmax" in nk or "r_max" in nk:
            score += 30
        if "ratio" in nk:
            score += 20
        if "enhancement" in nk:
            score += 20
        if "mixed_max" in nk:
            score += 15

        if score > best_score:
            best_score = score
            best_key = key

    return values[best_key] if best_key is not None and best_score > 0 else None


def metric_values_for_entry(entry: CrossingEntry, metric: str) -> tuple[Any, Any, Any, Any, Any]:
    v = entry.values

    e1 = find_metric_value(v, metric, "E1")
    e2 = find_metric_value(v, metric, "E2")
    ep = find_metric_value(v, metric, "Eplus")
    em = find_metric_value(v, metric, "Eminus")
    r = find_ratio_value(v, metric)

    # Homotypic CSV fallback: the CSV file/class already defines the metric.
    if entry.family == "homotypic" and metric in APPENDIX_CLASSES[entry.class_key]["rows"]:
        if e1 is None:
            e1 = find_generic_state_value(v, "E1")
        if e2 is None:
            e2 = find_generic_state_value(v, "E2")
        if ep is None:
            ep = find_generic_state_value(v, "Eplus")
        if em is None:
            em = find_generic_state_value(v, "Eminus")
        if r is None:
            r = find_ratio_value(v, None)

    if r is None:
        parents = [abs(x) for x in (to_float(e1), to_float(e2)) if x is not None]
        mixed = [abs(x) for x in (to_float(ep), to_float(em)) if x is not None]
        if parents and mixed and max(parents) != 0:
            r = max(mixed) / max(parents)

    return e1, e2, ep, em, r


def record_has_any_metric(rec: dict[str, Any]) -> bool:
    dummy = CrossingEntry(family="heterotypic", class_key="mono_dipole", values=rec)
    for metric in ["loss", "kick", "Kxx", "Kyy", "Kxy"]:
        vals = metric_values_for_entry(dummy, metric)
        if any(v is not None for v in vals[:4]):
            return True
    return False


def set_default_mode_labels(record: dict[str, Any]) -> dict[str, Any]:
    m1 = compact_mode_label(get_value(record, "mode_1"))
    m2 = compact_mode_label(get_value(record, "mode_2"))
    if m1:
        record["mode_1"] = m1
    if m2:
        record["mode_2"] = m2
    return record


def merge_records_by_identity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for rec in records:
        rec = set_default_mode_labels(rec)

        m1 = compact_mode_label(get_value(rec, "mode_1"))
        m2 = compact_mode_label(get_value(rec, "mode_2"))
        ell = to_float(get_value(rec, "ell"))
        f_hat = to_float(get_value(rec, "f_hat"))

        key = "|".join([
            m1 or "?",
            m2 or "?",
            f"{ell:.8g}" if ell is not None else "?",
            f"{f_hat:.8g}" if f_hat is not None else "?",
        ])

        if key not in merged:
            merged[key] = {}

        for k, v in rec.items():
            if is_missing_or_array_like(v):
                continue
            if k not in merged[key] or is_missing_or_array_like(merged[key][k]):
                merged[key][k] = v

    return list(merged.values())


def extract_records(obj: Any) -> list[dict[str, Any]]:
    raw_records: list[dict[str, Any]] = []

    def visit(x: Any) -> None:
        if isinstance(x, dict):
            flat = flatten(x)
            keys = {normal_key(k) for k in flat}

            has_mode = any(normal_key(a) in keys for a in KEY_ALIASES["mode_1"]) and \
                       any(normal_key(a) in keys for a in KEY_ALIASES["mode_2"])

            has_metric_name = any(
                any(normal_key(p) in key or key in normal_key(p) for p in spec["patterns"])
                for key in keys
                for spec in METRIC_SPECS.values()
            )

            if has_mode or has_metric_name:
                raw_records.append(flat)

            for v in x.values():
                if isinstance(v, (dict, list, tuple)):
                    visit(v)

        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)

    visit(obj)

    records = merge_records_by_identity(raw_records)
    with_modes = [
        set_default_mode_labels(r)
        for r in records
        if get_value(r, "mode_1") is not None and get_value(r, "mode_2") is not None
    ]

    metric_records = [r for r in with_modes if record_has_any_metric(r)]
    return metric_records if metric_records else with_modes


def metric_row_for_entry(entry: CrossingEntry, metric: str) -> str:
    spec = METRIC_SPECS[metric]
    e1_raw, e2_raw, ep_raw, em_raw, r_raw = metric_values_for_entry(entry, metric)

    return (
        f"{spec['label']} & {spec['units']} & "
        f"${fmt_num(e1_raw)}$ & ${fmt_num(e2_raw)}$ & "
        f"${fmt_num(ep_raw)}$ & ${fmt_num(em_raw)}$ & {fmt_ratio(r_raw)} \\\\"
    )


def mode_family(mode: Any) -> str | None:
    label = compact_mode_label(mode)
    if not label:
        return None
    m = re.search(r"TM(\d)(\d)(\d)", label, flags=re.IGNORECASE)
    if not m:
        return None
    az = m.group(1)
    if az == "0":
        return "mono"
    if az == "1":
        return "dipole"
    if az == "2":
        return "quad"
    return None


def heterotypic_class_key(record: dict[str, Any]) -> str | None:
    f1 = mode_family(get_value(record, "mode_1"))
    f2 = mode_family(get_value(record, "mode_2"))
    pair = {f1, f2}

    if pair == {"mono", "dipole"}:
        return "mono_dipole"
    if pair == {"mono", "quad"}:
        return "mono_quad"
    if pair == {"dipole", "quad"}:
        return "dipole_quad"
    return None


def entry_sort_key(entry: CrossingEntry) -> tuple[str, float, str]:
    mode_1 = compact_mode_label(get_value(entry.values, "mode_1")) or ""
    mode_2 = compact_mode_label(get_value(entry.values, "mode_2")) or ""
    ell = to_float(get_value(entry.values, "ell"))
    return mode_1, ell if ell is not None else -1.0, mode_2


def print_metric_diagnostics(label: str, rows: list[dict[str, Any]], class_key: str | None = None) -> None:
    if not rows:
        return

    row = set_default_mode_labels(dict(rows[0]))
    print(f"  diagnostics for {label}, first row:")
    print("    mode_1:", get_value(row, "mode_1"), "mode_2:", get_value(row, "mode_2"))
    print("    first 40 keys:", list(row.keys())[:40])

    if class_key is None:
        dummy = CrossingEntry(family="heterotypic", class_key="mono_dipole", values=row)
        metrics = ["loss", "kick", "Kxx", "Kyy", "Kxy"]
    else:
        dummy = CrossingEntry(
            family=APPENDIX_CLASSES[class_key]["family"],
            class_key=class_key,
            values=row,
        )
        metrics = APPENDIX_CLASSES[class_key]["rows"]

    for metric in metrics:
        print(f"    {metric}:", metric_values_for_entry(dummy, metric))


def collect_entries(root: Path) -> dict[str, list[CrossingEntry]]:
    grouped: dict[str, list[CrossingEntry]] = {key: [] for key in APPENDIX_CLASSES}

    # Homotypic CSV rows.
    for class_key, meta in APPENDIX_CLASSES.items():
        if meta["family"] != "homotypic":
            continue

        csv_path = root / meta["csv_rel"]
        rows = read_csv_dicts(csv_path)
        print(f"Loaded {len(rows)} rows: {csv_path}")
        print_metric_diagnostics(class_key, rows, class_key=class_key)

        for row in rows:
            row = set_default_mode_labels(row)
            grouped[class_key].append(
                CrossingEntry(family="homotypic", class_key=class_key, values=row)
            )

    # Heterotypic PKL records.
    pkl_path = root / HETEROTYPIC_PKL_REL
    obj = load_pickle(pkl_path)
    hetero_records = extract_records(obj) if obj is not None else []
    print(f"Loaded {len(hetero_records)} heterotypic records: {pkl_path}")
    print_metric_diagnostics("heterotypic", hetero_records, class_key=None)

    unclassified = 0
    for rec in hetero_records:
        rec = set_default_mode_labels(rec)
        class_key = heterotypic_class_key(rec)
        if class_key is None:
            unclassified += 1
            continue

        grouped[class_key].append(
            CrossingEntry(family="heterotypic", class_key=class_key, values=rec)
        )

    if unclassified:
        print(f"[warning] Heterotypic records not classifiable by TM azimuthal index: {unclassified}")

    for key in grouped:
        grouped[key] = sorted(grouped[key], key=entry_sort_key)

    return grouped


def title_line(entry: CrossingEntry) -> str:
    meta = APPENDIX_CLASSES[entry.class_key]
    mode_1 = get_value(entry.values, "mode_1")
    mode_2 = get_value(entry.values, "mode_2")
    ell = get_value(entry.values, "ell")
    f_hat = get_value(entry.values, "f_hat")

    items = [
        rf"\textbf{{{meta['display']} crossing "
        rf"{latex_mode_label(mode_1)}--{latex_mode_label(mode_2)}",
    ]

    if ell is not None:
        items.append(rf"$\ell={fmt_num(ell, sig=4)}$")
    if f_hat is not None:
        items.append(rf"$\hat{{f}}={fmt_num(f_hat, sig=4)}$")

    return ", ".join(items) + r"}\\[0.5em]"


def metric_table(entry: CrossingEntry) -> str:
    rows_to_show = APPENDIX_CLASSES[entry.class_key]["rows"]
    rows = "\n".join(metric_row_for_entry(entry, metric) for metric in rows_to_show)

    return rf"""
\begin{{center}}
\small
{title_line(entry)}
\begin{{ruledtabular}}
\begin{{tabular}}{{ccccccc}}
Effect & Units & $E_1$ & $E_2$ & $E_+$ & $E_-$ & $R_{{\max}}$ \\
\hline
{rows}
\end{{tabular}}
\end{{ruledtabular}}
\end{{center}}
""".strip()


def plots_block() -> str:
    p = TEST_PLOT_FILENAMES
    return rf"""
\begin{{center}}
\begin{{tabular}}{{cc}}
\includegraphics[width=0.44\textwidth]{{{p["iris_1"]}}} &
\includegraphics[width=0.44\textwidth]{{{p["iris_2"]}}} \\
\includegraphics[width=0.44\textwidth]{{{p["longitudinal_mid"]}}} &
\includegraphics[width=0.44\textwidth]{{{p["transverse_mid"]}}}
\end{{tabular}}
\end{{center}}
""".strip()


def crossing_block(entry: CrossingEntry) -> str:
    return "\n".join([
        metric_table(entry),
        r"\vspace{-1.0em}",
        plots_block(),
        r"\vspace{-0.5em}",
    ])


def latex_appendices(grouped: dict[str, list[CrossingEntry]]) -> str:
    parts: list[str] = []

    parts.append(r"\appendix")
    parts.append("")
    parts.append(r"\section{Appendix I: Homotypic Mode Mixing}")
    parts.append("")

    for class_key in ["mono_mono", "dipole_dipole", "quad_quad"]:
        parts.append(rf"\subsection{{{APPENDIX_CLASSES[class_key]['title']}}}")
        parts.append("")

        entries = grouped.get(class_key, [])
        if not entries:
            parts.append(rf"% No entries found for {APPENDIX_CLASSES[class_key]['title']}.")
            parts.append("")
            continue

        for entry in entries:
            parts.append(crossing_block(entry))
            parts.append(r"\clearpage")
            parts.append("")

    parts.append(r"\section{Appendix II: Heterotypic Mode Mixing}")
    parts.append("")

    for class_key in ["mono_dipole", "mono_quad", "dipole_quad"]:
        parts.append(rf"\subsection{{{APPENDIX_CLASSES[class_key]['title']}}}")
        parts.append("")

        entries = grouped.get(class_key, [])
        if not entries:
            parts.append(rf"% No entries found for {APPENDIX_CLASSES[class_key]['title']}.")
            parts.append("")
            continue

        for entry in entries:
            parts.append(crossing_block(entry))
            parts.append(r"\clearpage")
            parts.append("")

    return "\n".join(parts).strip() + "\n"


def main(root: Path | str | None = None, out: Path | str | None = None) -> None:
    """
    ROOT_DIR is the root of the stored analysis data, not this script location.
    """

    # ------------------------------------------------------------------
    # EDIT THESE TWO PATHS
    # ------------------------------------------------------------------
    ROOT_DIR = Path(
        r"D:\PhD\HOMmix\HOMmix_analytical\analysis"
    )

    OUT_TEX = Path(
        r"D:\PhD\PRAB\mode_mixing_appendices.tex"
    )
    # ------------------------------------------------------------------

    root_path = Path(root) if root is not None else ROOT_DIR
    out_path = Path(out) if out is not None else OUT_TEX

    root_path = root_path.resolve()

    print()
    print("Script location:")
    print(Path(__file__).resolve())
    print()
    print("Current working directory:")
    print(Path.cwd())
    print()
    print("ROOT_DIR being searched:")
    print(root_path)
    print()

    grouped = collect_entries(root_path)
    tex = latex_appendices(grouped)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex, encoding="utf-8")

    print()
    print(f"Wrote: {out_path}")
    print()
    for key, meta in APPENDIX_CLASSES.items():
        print(f"{meta['title']}: {len(grouped.get(key, []))} entries")


if __name__ == "__main__":
    main()
