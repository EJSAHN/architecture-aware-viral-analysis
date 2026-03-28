from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_panel_registry(curated_df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = curated_df.copy()
    min_mono = int(config["analysis"]["min_group_size_monopartite"])

    mono_counts = df.loc[df["schema_type"] == "monopartite_genome", "source_group"].value_counts()
    eligible_monopartite_groups = set(mono_counts[mono_counts >= min_mono].index.tolist())

    df["panel_all_curated"] = True
    df["panel_monopartite"] = df["schema_type"].eq("monopartite_genome") & df["source_group"].isin(eligible_monopartite_groups)
    df["panel_componentized"] = df["schema_type"].isin(["bipartite_component", "multipartite_component"]) & df["normalized_component"].notna()

    df["analysis_unit"] = df.apply(
        lambda row: (
            f"{row['source_group']}|{row['normalized_component']}"
            if row["panel_componentized"]
            else str(row["source_group"])
        ),
        axis=1,
    )

    panel_rows = [
        {
            "panel_name": "all_curated",
            "n_records": int(df["panel_all_curated"].sum()),
            "n_groups": int(df["analysis_unit"].nunique()),
            "group_labels": "; ".join(sorted(df["analysis_unit"].astype(str).unique().tolist())),
        },
        {
            "panel_name": "monopartite",
            "n_records": int(df["panel_monopartite"].sum()),
            "n_groups": int(df.loc[df["panel_monopartite"], "source_group"].nunique()),
            "group_labels": "; ".join(sorted(df.loc[df["panel_monopartite"], "source_group"].astype(str).unique().tolist())),
        },
        {
            "panel_name": "componentized",
            "n_records": int(df["panel_componentized"].sum()),
            "n_groups": int(df.loc[df["panel_componentized"], ["source_group", "normalized_component"]].drop_duplicates().shape[0]),
            "group_labels": "; ".join(
                sorted(
                    df.loc[df["panel_componentized"]]
                    .apply(lambda row: f"{row['source_group']}|{row['normalized_component']}", axis=1)
                    .unique()
                    .tolist()
                )
            ),
        },
    ]

    per_group = (
        df.groupby(["source_group", "schema_type", "normalized_component"], dropna=False)
        .size()
        .reset_index(name="n_records")
        .sort_values(["source_group", "schema_type", "normalized_component"], ignore_index=True)
    )
    return df, pd.DataFrame(panel_rows), per_group
