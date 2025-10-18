import argparse
import os
import shutil
import time

from typing import Dict, List, Tuple

import pandas as pd


def list_files_with_prefix(directory: str, prefix: str, suffix: str) -> List[str]:
    return [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(suffix)]


def read_metadata_parts(output_dir: str) -> List[pd.DataFrame]:
    parts: List[pd.DataFrame] = []
    for f in os.listdir(output_dir):
        if f.startswith("metadata_") and f.endswith(".csv"):
            try:
                parts.append(pd.read_csv(os.path.join(output_dir, f), low_memory=False))
            except Exception:
                pass
    return parts


def normalize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure unique by sha256 and set index
    if "sha256" in df.columns:
        df = df.drop_duplicates(subset=["sha256"], keep="last")
        df.set_index("sha256", inplace=True)
    elif df.index.name == "sha256":
        df = df[~df.index.duplicated(keep="last")]

    # Known boolean/process columns used across toolkits
    required_bool_cols = [
        "rendered",
        "fixview_rendered",
        "cond_rendered",
        "cond_rendered_test",
        "voxelized",
    ]
    for col in required_bool_cols:
        if col not in df.columns:
            df[col] = [False] * len(df)

    # num_voxels as int
    if "num_voxels" not in df.columns:
        df["num_voxels"] = [0] * len(df)

    # Convert obvious flags to boolean if present as 0/1 or strings
    for col in df.columns:
        if (
            col in required_bool_cols
            or col.startswith("feature_")
            or col.startswith("latent_")
            or col.startswith("ss_latent_")
        ):
            if df[col].dtype not in [bool, "bool"]:
                try:
                    df[col] = df[col].astype(bool)
                except Exception:
                    pass
    return df


def merge_boolean_columns(base: pd.DataFrame, other: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        if col not in base.columns and col in other.columns:
            base[col] = False
        if col in other.columns:
            # OR semantics: if any source marks True, keep True
            s = other[col].reindex(base.index)
            # Avoid FutureWarning on object downcasting by using where instead of fillna
            s = s.where(s.notna(), False)
            base[col] = base[col] | s


def merge_metadata_frames(parts: List[pd.DataFrame]) -> pd.DataFrame:
    if len(parts) == 0:
        raise FileNotFoundError("No metadata_*.csv parts found to merge.")

    # Normalize all first
    parts = [normalize_metadata_columns(p) for p in parts]

    # Union of indices
    all_index = parts[0].index
    for p in parts[1:]:
        all_index = all_index.union(p.index)

    # Start with empty frame indexed by union
    merged = pd.DataFrame(index=all_index)

    # Collect all columns
    all_cols: List[str] = []
    for p in parts:
        for c in p.columns:
            if c not in all_cols:
                all_cols.append(c)
    for c in all_cols:
        merged[c] = None

    # Prefer later parts for non-boolean, union-or for boolean-ish columns
    bool_like = [
        "rendered",
        "fixview_rendered",
        "cond_rendered",
        "cond_rendered_test",
        "voxelized",
    ]
    # Also treat model flags as boolean
    bool_like += [
        c for c in all_cols if c.startswith("feature_") or c.startswith("latent_") or c.startswith("ss_latent_")
    ]

    # Initialize boolean columns as False
    for c in bool_like:
        if c in merged.columns:
            merged[c] = False

    # Merge
    for p in parts:
        # Booleans: OR
        present_bool_cols = [c for c in bool_like if c in p.columns]
        merge_boolean_columns(merged, p, present_bool_cols)

        # Non-booleans: prefer non-null values in later parts
        for c in p.columns:
            if c in bool_like:
                continue
            # Fill only where currently missing; do not overwrite existing non-null values
            merged[c] = merged[c].where(merged[c].notna(), p[c].reindex(merged.index))

    # Final tidy types
    if "num_voxels" in merged.columns:
        try:
            col = merged["num_voxels"]
            col = col.where(col.notna(), 0)
            merged["num_voxels"] = col.astype(int)
        except Exception:
            pass

    # For string-like/object columns, fill missing with empty string per user request
    object_cols = [c for c in merged.columns if merged[c].dtype == object and c != "sha256"]
    for c in object_cols:
        merged[c] = merged[c].where(merged[c].notna(), "")

    # Reset index to include sha256 column
    merged = merged.reset_index()
    return merged


def archive_source_files(output_dir: str, files: List[str], subfolder: str, timestamp: str) -> None:
    os.makedirs(os.path.join(output_dir, "merged_records"), exist_ok=True)
    for f in files:
        src = os.path.join(output_dir, f)
        if os.path.exists(src):
            dst = os.path.join(output_dir, "merged_records", f"{timestamp}_{f}")
            try:
                shutil.move(src, dst)
            except Exception:
                pass


def read_statistics_parts(output_dir: str) -> List[Tuple[str, List[str]]]:
    stats_files = [f for f in os.listdir(output_dir) if f.startswith("statistics_") and f.endswith(".txt")]
    parts: List[Tuple[str, List[str]]] = []
    for f in stats_files:
        try:
            with open(os.path.join(output_dir, f), "r") as fp:
                parts.append((f, fp.readlines()))
        except Exception:
            pass
    return parts


def aggregate_statistics_lines(parts: List[Tuple[str, List[str]]], merged_meta: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    # Header must match part files exactly
    lines.append("Statistics:\n")

    # Compute global statistics based on merged metadata
    num_assets = len(merged_meta)

    def count_non_empty(series_name: str) -> int:
        if series_name not in merged_meta.columns:
            return 0
        s = merged_meta[series_name]
        mask = s.notna() & (s.astype(str) != "")
        return int(mask.sum())

    def sum_flag(flag_name: str) -> int:
        if flag_name not in merged_meta.columns:
            return 0
        s = merged_meta[flag_name]
        try:
            return int(pd.Series(s).astype(bool).sum())
        except Exception:
            return int(pd.Series(s).astype(bool).sum())

    num_downloaded = count_non_empty("local_path")
    num_rendered = sum_flag("rendered")
    num_voxelized = sum_flag("voxelized")

    lines.append(f"  - Number of assets: {num_assets}\n")
    lines.append(f"  - Number of assets downloaded: {num_downloaded}\n")
    lines.append(f"  - Number of assets rendered: {num_rendered}\n")
    lines.append(f"  - Number of assets voxelized: {num_voxelized}\n")

    # Per-model counts using exact wording from part files
    model_cols = [c for c in merged_meta.columns if c.startswith("feature_")]
    if len(model_cols) != 0:
        lines.append("  - Number of assets with image features extracted:\n")
        for c in model_cols:
            lines.append(f"    - {c[len('feature_') :]}: {int(merged_meta[c].sum())}\n")

    latent_cols = [c for c in merged_meta.columns if c.startswith("latent_")]
    if len(latent_cols) != 0:
        lines.append("  - Number of assets with latents extracted:\n")
        for c in latent_cols:
            lines.append(f"    - {c[len('latent_') :]}: {int(merged_meta[c].sum())}\n")

    ss_latent_cols = [c for c in merged_meta.columns if c.startswith("ss_latent_")]
    if len(ss_latent_cols) != 0:
        lines.append("  - Number of assets with sparse structure latents extracted:\n")
        for c in ss_latent_cols:
            lines.append(f"    - {c[len('ss_latent_') :]}: {int(merged_meta[c].sum())}\n")

    # Additional lines seen in part files
    lines.append(f"  - Number of assets with captions: {count_non_empty('captions')}\n")
    lines.append(f"  - Number of assets with fixview rendered: {sum_flag('fixview_rendered')}\n")
    lines.append(f"  - Number of assets with image conditions: {sum_flag('cond_rendered')}\n")
    lines.append(f"  - Number of assets with test image conditions: {sum_flag('cond_rendered_test')}\n")

    # Append a section enumerating full source stats files
    if len(parts) != 0:
        lines.append("\nSources merged (raw snippets):\n")
        for fname, content in parts:
            lines.append(f"source: {fname}\n")
            # Include full content as-is
            lines.extend(content)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Merged dataset directory to write outputs")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = str(int(time.time()))

    # 1) Load and merge metadata parts
    parts = read_metadata_parts(output_dir)
    if len(parts) == 0:
        raise SystemExit("No metadata_*.csv found under the merged folder. Nothing to merge.")

    merged = merge_metadata_frames(parts)

    # 2) Write merged metadata
    merged_path = os.path.join(output_dir, "metadata.csv")
    merged.to_csv(merged_path, index=False)

    # 3) Archive source metadata files
    meta_files = [f for f in os.listdir(output_dir) if f.startswith("metadata_") and f.endswith(".csv")]
    archive_source_files(output_dir, meta_files, "merged_records", timestamp)

    # 4) Merge statistics text summaries into a new synthesized statistics.txt
    stat_parts = read_statistics_parts(output_dir)
    stat_lines = aggregate_statistics_lines(stat_parts, merged)
    with open(os.path.join(output_dir, "statistics.txt"), "w") as fp:
        fp.writelines(stat_lines)

    # 5) Archive source statistics files (statistics_*.txt)
    stat_files = [f for f, _ in stat_parts]
    archive_source_files(output_dir, stat_files, "merged_records", timestamp)


if __name__ == "__main__":
    main()
