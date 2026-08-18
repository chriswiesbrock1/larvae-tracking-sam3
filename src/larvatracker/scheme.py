"""Mapping droplet IDs to experimental groups.

Which droplet received which treatment is recorded by hand in a scheme file
next to the recording. The file needs two columns:

===========  ==============================================
``Droplet``  integer droplet ID, matching ``droplets.csv``
``Group``    free-text treatment label, e.g. ``Asp 5``
===========  ==============================================

Both ``.xlsx`` and ``.csv`` are accepted. See ``examples/scheme_template.csv``.
"""

from __future__ import annotations

import os
import re

import pandas as pd

DROPLET_ID_PATTERN = re.compile(r"droplet_(\d+)")

UNKNOWN_GROUP = "Unknown"


def load_scheme(path: str) -> pd.DataFrame:
    """Read a scheme file and normalise its two required columns.

    CSV scheme files are written by hand and turn up with either a comma or a
    semicolon separator depending on the machine's locale — a German Excel
    exports semicolons. The separator is therefore sniffed rather than assumed;
    guessing wrong would produce a single merged column and a confusing
    "missing column" error.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

        if len(df.columns) == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(path, sep=";")

    missing = {"Droplet", "Group"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {sorted(missing)}. "
            "Expected columns 'Droplet' and 'Group'."
        )

    df = df.dropna(subset=["Droplet"]).copy()
    df["Droplet"] = df["Droplet"].astype(int)
    df["Group"] = df["Group"].astype(str).str.strip()
    return df


def droplet_id_from_filename(filename: str) -> int | None:
    """Extract the droplet ID from a DeepLabCut output filename.

    DeepLabCut appends the network name to the video name, so the files are
    called e.g. ``droplet_007DLC_Resnet50_...csv``. Returns None when the name
    does not follow that convention.
    """
    match = DROPLET_ID_PATTERN.search(os.path.basename(filename))
    return int(match.group(1)) if match else None


def group_for_droplet(scheme: pd.DataFrame | None, droplet_id: int) -> str:
    """Look up the treatment group of a droplet, defaulting to ``Unknown``."""
    if scheme is None:
        return UNKNOWN_GROUP

    match = scheme.loc[scheme["Droplet"] == droplet_id, "Group"]
    return str(match.values[0]) if not match.empty else UNKNOWN_GROUP


def normalise_group_labels(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Fold inconsistent hand-typed group labels onto canonical names.

    Labels are lower-cased and whitespace-collapsed before lookup, so
    ``"Asp  5"``, ``"asp 5"`` and ``"ASP 5"`` all map to the same key. Values
    with no entry in ``mapping`` become NaN, which makes unmapped labels easy
    to spot instead of silently forming their own group.
    """
    key = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    return key.map(mapping)
