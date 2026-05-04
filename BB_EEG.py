# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:33:01 2026

@author: msedo
"""

from pathlib import Path
import re
import sqlite3
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --------------------------------------------------
# Paths
# --------------------------------------------------
edf_dir = Path(r'C:\Users\marti\OneDrive\Documents\HSJD\Beating Brains\CCV_EEG\2068883_18_11_2025')
db_path = Path(r"C:\Users\marti\OneDrive\Documents\HSJD\Beating Brains\CCV_EEG\2068883_18_11_2025\patient.db")

# Excel file
file_path = r"C:\Users\marti\OneDrive\Documents\HSJD\Beating Brains\UNIFIED_5MIN_SYNCHRONIZED.xlsx"

def ensure_utc_timestamp(x):
    """Return a pandas Timestamp that is always UTC-aware."""
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def parse_dt_from_filename(name: str):
    m = re.match(r"(\d{8})_(\d{6})_(\d{3})\.", name)
    if not m:
        return pd.NaT
    d, t, ms = m.groups()
    ts = pd.to_datetime(f"{d}{t}{ms}", format="%Y%m%d%H%M%S%f")
    return ts.tz_localize("UTC")

# -----------------------------
# Read DB comments
# -----------------------------
conn = sqlite3.connect(db_path)

comments = pd.read_sql_query("""
SELECT
    c.Id AS comment_id,
    c.CommentDateTime AS ts100ns,
    ct.CommentText AS label
FROM Comment c
LEFT JOIN CommentText ct ON ct.Id = c.CommentTextId
ORDER BY c.CommentDateTime
""", conn)

comments["ts100ns"] = np.rint(comments["ts100ns"]).astype("int64")

# This is the important line:
comments["dt"] = pd.to_datetime(
    comments["ts100ns"] * 100,
    unit="ns",
    origin="unix",
    utc=True,   # force UTC-aware
)

comments["label"] = comments["label"].fillna("").astype(str)

print(comments[["dt", "label"]])
print(comments["dt"].dtype)   # should show: datetime64[ns, UTC]



# -----------------------------
# Loop EDFs
# -----------------------------
edf_files = sorted(edf_dir.glob("*.edf"))
annotated_raws = []

for fp in edf_files:
    raw = mne.io.read_raw_edf(fp, preload=True, verbose="ERROR")

    meas_date = raw.info.get("meas_date")
    if meas_date is not None:
        start_dt = ensure_utc_timestamp(meas_date)
    else:
        start_dt = parse_dt_from_filename(fp.name)

    if pd.isna(start_dt):
        print(f"Skipping {fp.name}: no valid start time")
        continue
    
    raw.set_meas_date(start_dt.to_pydatetime())

    duration_s = raw.n_times / raw.info["sfreq"]
    end_dt = start_dt + pd.to_timedelta(duration_s, unit="s")

    print(f"\n{fp.name}")
    print("start_dt:", start_dt, "| tz:", start_dt.tzinfo)
    print("comments dtype:", comments["dt"].dtype)

    sel = comments[(comments["dt"] >= start_dt) & (comments["dt"] <= end_dt)].copy()

    if not sel.empty:
        onsets = (sel["dt"] - start_dt).dt.total_seconds().to_numpy()

        ann = mne.Annotations(
            onset=onsets,
            duration=np.zeros(len(sel), dtype=float),
            description=sel["label"].tolist(),
            orig_time=start_dt.to_pydatetime(),
        )

        raw.set_annotations(raw.annotations + ann if len(raw.annotations) else ann)

    print(f"Annotations added: {len(sel)}")
    annotated_raws.append((fp, raw, start_dt, end_dt))

# --------------------------------------------------
# Plot each EDF with real clock time
# --------------------------------------------------

win_start = ensure_utc_timestamp("2025-11-18 10:00:00")
win_end = ensure_utc_timestamp("2025-11-18 15:30:00")

# for fp, raw, start_dt, end_dt in annotated_raws:
#     if start_dt <= win_end and end_dt >= win_start:
#      #if start_dt <= win_end:   
#         raw.filter(l_freq=0.5, h_freq=30)
#         raw.plot(
#             duration=8,
#             n_channels=min(30, len(raw.ch_names)),
#             scalings="400e-6",
#             time_format="datetime",
#             title=f"{fp.name} | {start_dt.strftime('%H:%M:%S')} - {end_dt.strftime('%H:%M:%S')}",
#             show=True,
#             block=True,
#         )



# --------------------------------------------------
# Read intraoperative variables from Excel
# --------------------------------------------------
# --------------------------------------------------
# Excel variables + plotting configuration
# --------------------------------------------------
from zoneinfo import ZoneInfo

# --------------------------------------------------
# Excel variables + plotting configuration
# --------------------------------------------------

EXCEL_SOURCE_TZ = "UTC"      # Excel HH:MM times are interpreted as UTC
DISPLAY_TZ = "UTC"           # Plot x-axis is displayed in UTC

TIME_COL = "Time"            # Change if your Excel time column has another name
NORMALIZE_VARIABLES = True  # True = z-score normalization, False = raw values

# Optional: set explicitly if needed
# If None, the date is inferred from the first EEG annotation date
SURGERY_DATE_LOCAL = None    # Example: "2025-11-18"


# --------------------------------------------------
# Optional manual plotting window
# Leave as None to use automatic first/last Excel timestamp
# Manual times are interpreted in EXCEL_SOURCE_TZ unless timezone-aware
# --------------------------------------------------

MANUAL_WIN_START = "2025-11-18 10:30:00"
MANUAL_WIN_END = "2025-11-18 12:30:00"

# MANUAL_WIN_START = None
# MANUAL_WIN_END = None

def parse_excel_clock_column(s):
    """
    Robustly parse an Excel time column that may contain:
    - strings like '11:10' or '11:10:30'
    - datetime/time objects
    - Excel fractional days
    """
    out = []

    for x in s:
        if pd.isna(x):
            out.append(pd.NaT)
            continue

        if isinstance(x, pd.Timestamp):
            out.append(x.time())
            continue

        if hasattr(x, "time") and not isinstance(x, str):
            try:
                out.append(x.time())
                continue
            except Exception:
                pass

        if isinstance(x, (int, float, np.integer, np.floating)):
            seconds = int(round(float(x) * 24 * 3600))
            seconds = seconds % (24 * 3600)
            out.append(
                (pd.Timestamp("1900-01-01") + pd.to_timedelta(seconds, unit="s")).time()
            )
            continue

        txt = str(x).strip()
        parsed = pd.to_datetime(txt, errors="coerce")

        if pd.isna(parsed):
            out.append(pd.NaT)
        else:
            out.append(parsed.time())

    return pd.Series(out, index=s.index)


def attach_date_and_convert_to_utc(df, time_col, base_date, source_tz):
    """
    Attach a calendar date to HH:MM(:SS) times and convert to UTC.

    Since EXCEL_SOURCE_TZ is now UTC, Excel clock times such as 09:05
    are interpreted as 09:05 UTC.
    """
    df = df.copy()

    clock_times = parse_excel_clock_column(df[time_col])
    df = df.loc[clock_times.notna()].copy()
    clock_times = clock_times.loc[df.index]

    local_datetimes = [
        pd.Timestamp.combine(pd.Timestamp(base_date).date(), t)
        for t in clock_times
    ]

    df["dt_local"] = pd.to_datetime(local_datetimes)

    df["dt"] = df["dt_local"].dt.tz_localize(source_tz).dt.tz_convert("UTC")

    # If the case crosses midnight, fix rows where time appears to go backwards
    day_jumps = df["dt"].diff().dt.total_seconds().lt(-12 * 3600).cumsum()
    df["dt"] = df["dt"] + pd.to_timedelta(day_jumps, unit="D")
    df["dt_local"] = df["dt"].dt.tz_convert(source_tz)

    return df


def local_or_aware_to_utc(x, source_tz, base_date=None):
    """
    Convert a manual timestamp to UTC.

    If x is timezone-naive, it is interpreted as source_tz.
    Since EXCEL_SOURCE_TZ is now UTC, manual times such as
    '2025-11-18 10:30:00' are interpreted as 10:30 UTC.

    Allows:
    - "2025-11-18 10:30:00"
    - "10:30"
    - "10:30:00"
    """
    if x is None:
        return None

    if isinstance(x, str):
        txt = x.strip()

        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", txt):
            if base_date is None:
                raise ValueError(
                    "Time-only manual windows require base_date. "
                    "Use a full datetime or provide base_date."
                )
            txt = f"{pd.Timestamp(base_date).date()} {txt}"

        ts = pd.Timestamp(txt)
    else:
        ts = pd.Timestamp(x)

    if ts.tzinfo is None:
        return ts.tz_localize(source_tz).tz_convert("UTC")

    return ts.tz_convert("UTC")


def get_plot_window(
    vars_df,
    manual_start=None,
    manual_end=None,
    manual_tz="UTC",
    base_date=None,
):
    """
    If manual_start/manual_end are provided, interpret timezone-naive values
    as manual_tz.

    Otherwise, default to the first and last timestamp in vars_df.
    """
    auto_start = vars_df["dt"].min()
    auto_end = vars_df["dt"].max()

    win_start = (
        local_or_aware_to_utc(manual_start, manual_tz, base_date=base_date)
        if manual_start is not None
        else auto_start
    )

    win_end = (
        local_or_aware_to_utc(manual_end, manual_tz, base_date=base_date)
        if manual_end is not None
        else auto_end
    )

    return win_start, win_end


# --------------------------------------------------
# Determine surgery date for Excel HH:MM timestamps
# --------------------------------------------------

if SURGERY_DATE_LOCAL is not None:
    base_date_for_excel = pd.Timestamp(SURGERY_DATE_LOCAL).date()
else:
    # Infer date from the first EEG annotation
    base_date_for_excel = comments["dt"].min().tz_convert(EXCEL_SOURCE_TZ).date()

print("Base date used for Excel timestamps:", base_date_for_excel)


# --------------------------------------------------
# Read Excel and create UTC-aware timestamps
# --------------------------------------------------

vars_df = pd.read_excel(file_path)

vars_df = attach_date_and_convert_to_utc(
    vars_df,
    time_col=TIME_COL,
    base_date=base_date_for_excel,
    source_tz=EXCEL_SOURCE_TZ,
)


# --------------------------------------------------
# Define plotting window
# --------------------------------------------------

win_start, win_end = get_plot_window(
    vars_df,
    manual_start=MANUAL_WIN_START,
    manual_end=MANUAL_WIN_END,
    manual_tz=EXCEL_SOURCE_TZ,
    base_date=base_date_for_excel,
)

print("Plotting window in UTC:")
print("win_start:", win_start)
print("win_end:", win_end)

print(f"\nPlotting window shown on figure ({DISPLAY_TZ}):")
print("win_start:", win_start.tz_convert(DISPLAY_TZ))
print("win_end:", win_end.tz_convert(DISPLAY_TZ))


# --------------------------------------------------
# Keep numeric variables only
# --------------------------------------------------

variable_cols = [
    c for c in vars_df.columns
    if c not in [TIME_COL, "dt", "dt_local"]
]

for c in variable_cols:
    vars_df[c] = pd.to_numeric(vars_df[c], errors="coerce")

variable_cols = [
    c for c in variable_cols
    if vars_df[c].notna().sum() > 0
]

print("\nVariable columns detected:")
print(variable_cols)

print(vars_df[["dt", "dt_local", TIME_COL] + variable_cols].head())
print(vars_df["dt"].dtype)


# --------------------------------------------------
# Plot variables + EEG annotation markers
# --------------------------------------------------

def zscore_series(s):
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(skipna=True)

    if pd.isna(std) or std == 0:
        return s * np.nan

    return (s - s.mean(skipna=True)) / std

def assign_annotation_label_levels(ann_df, min_separation_seconds=90, max_levels=4):
    """
    Assign a vertical label level to each annotation so that labels that are
    close in time are stacked vertically instead of overlapping.

    Parameters
    ----------
    ann_df : DataFrame
        Must contain a 'dt' column with timezone-aware timestamps.
    min_separation_seconds : float
        Two labels closer than this are considered overlapping.
    max_levels : int
        Maximum number of stacked vertical levels.

    Returns
    -------
    DataFrame
        Copy of ann_df with an extra column: 'label_level'
    """
    if ann_df.empty:
        ann_df = ann_df.copy()
        ann_df["label_level"] = []
        return ann_df

    ann_df = ann_df.sort_values("dt").copy()

    # Stores the most recent timestamp assigned to each vertical level
    last_time_per_level = []
    assigned_levels = []

    for t in ann_df["dt"]:
        assigned = False

        # Try to place this label in the first free level
        for level in range(len(last_time_per_level)):
            delta_s = (t - last_time_per_level[level]).total_seconds()
            if delta_s >= min_separation_seconds:
                assigned_levels.append(level)
                last_time_per_level[level] = t
                assigned = True
                break

        # If no free level found, create a new one if possible
        if not assigned:
            if len(last_time_per_level) < max_levels:
                last_time_per_level.append(t)
                assigned_levels.append(len(last_time_per_level) - 1)
            else:
                # Reuse the last level if all are occupied
                assigned_levels.append(max_levels - 1)
                last_time_per_level[max_levels - 1] = t

    ann_df["label_level"] = assigned_levels
    return ann_df

def plot_variables_with_eeg_annotations(
    vars_df,
    comments,
    win_start,
    win_end,
    variable_cols,
    normalize=True,
    show_annotation_labels=True,
    display_tz="UTC",
    label_min_separation_seconds=90,
    label_vertical_step=0.04,
    max_label_levels=4,
):
    """
    Plot all intraoperative variables on one shared time axis and overlay EEG annotations.

    Only annotations whose label is a single digit (0-9) are plotted.
    Each annotation type is plotted in a different color.

    Annotation labels that are close in time are vertically staggered
    to reduce overlap.
    """

    win_start = ensure_utc_timestamp(win_start)
    win_end = ensure_utc_timestamp(win_end)

    display_tzinfo = ZoneInfo(display_tz)

    plot_df = vars_df[
        (vars_df["dt"] >= win_start) &
        (vars_df["dt"] <= win_end)
    ].copy()

    ann_df = comments[
        (comments["dt"] >= win_start) &
        (comments["dt"] <= win_end)
    ].copy()

    if plot_df.empty:
        print("No Excel variable data found inside the selected time window.")
        return

    # Keep only single-digit numeric annotations
    ann_df["label"] = ann_df["label"].astype(str).str.strip()
    ann_df = ann_df[ann_df["label"].str.fullmatch(r"\d")].copy()

    if ann_df.empty:
        print("No single-digit EEG annotations found inside the selected time window.")

    # Assign vertical levels to labels that are close in time
    ann_df = assign_annotation_label_levels(
        ann_df,
        min_separation_seconds=label_min_separation_seconds,
        max_levels=max_label_levels,
    )

    fig, ax = plt.subplots(figsize=(18, 8))

    # --------------------------------
    # Plot variables
    # --------------------------------
    for col in variable_cols:
        y = plot_df[col]

        if normalize:
            y = zscore_series(y)

        ax.plot(
            plot_df["dt"],
            y,
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=col,
        )

    # --------------------------------
    # Assign one color per annotation type
    # --------------------------------
    ann_types = sorted(ann_df["label"].unique(), key=int) if not ann_df.empty else []

    cmap = plt.get_cmap("tab10", max(len(ann_types), 1))
    ann_color_map = {
        ann_type: cmap(i) for i, ann_type in enumerate(ann_types)
    }

    # --------------------------------
    # Plot EEG annotations
    # Only one legend entry per event type
    # --------------------------------
    used_ann_labels = set()

    for ann_type in ann_types:
        sub = ann_df[ann_df["label"] == ann_type]

        for _, row in sub.iterrows():
            line_label = f"EEG {ann_type}" if ann_type not in used_ann_labels else None

            ax.axvline(
                row["dt"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                color=ann_color_map[ann_type],
                label=line_label,
            )

            if show_annotation_labels:
                y_text = 1.01 + row["label_level"] * label_vertical_step

                ax.text(
                    row["dt"],
                    y_text,
                    row["label"],
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=ann_color_map[ann_type],
                    clip_on=False,
                )

            used_ann_labels.add(ann_type)

    # --------------------------------
    # X-axis formatting: tick every 10 minutes
    # --------------------------------
    ax.xaxis.set_major_locator(
        mdates.MinuteLocator(interval=10, tz=display_tzinfo)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M", tz=display_tzinfo)
    )

    ax.set_xlim(win_start, win_end)

    if normalize:
        ax.set_ylabel("Normalized variable value (z-score)")
    else:
        ax.set_ylabel("Raw variable value")

    ax.set_xlabel(f"Time ({display_tz})")

    ax.set_title(
        "Variables intraoperatorias y eventos EEG",
        pad=25
    )

    ax.grid(True, alpha=0.3)

    # --------------------------------
    # Remove repeated legend entries
    # --------------------------------
    handles, labels = ax.get_legend_handles_labels()

    unique = {}
    for h, l in zip(handles, labels):
        if l and l not in unique:
            unique[l] = h

    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    

plot_variables_with_eeg_annotations(
    vars_df=vars_df,
    comments=comments,
    win_start=win_start,
    win_end=win_end,
    variable_cols=variable_cols,
    normalize=NORMALIZE_VARIABLES,
    show_annotation_labels=True,
    display_tz=DISPLAY_TZ,
    label_min_separation_seconds=30,   # increase if labels still overlap
    label_vertical_step=0.02,          # increase for more vertical spacing
    max_label_levels=10,                # maximum stacked rows of labels
)



# --------------------------------------------------
# EEG event vs physiological variable analysis
# --------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# --------------------------------------------------
# Analysis parameters
# --------------------------------------------------

EVENTS_TO_IGNORE = ["0"]          # 0 is baseline
DUPLICATE_TOLERANCE_SECONDS = 1   # collapse repeated markers at nearly same timestamp

RESAMPLE_RULE = "1min"            # regular time grid for variables
PRE_WINDOW_MIN = 5                # baseline window before event onset
POST_WINDOW_MIN = 5               # recovery window after event end

MIN_POINTS_PER_WINDOW = 1         # minimum variable samples required per window
MIN_EVENTS_FOR_STATS = 3          # minimum number of events for Wilcoxon test


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def prepare_numeric_annotations(comments, ignore_labels=None):
    """
    Keep only single-digit EEG annotations and remove ignored labels.
    """
    if ignore_labels is None:
        ignore_labels = []

    ann = comments.copy()
    ann["label"] = ann["label"].astype(str).str.strip()

    ann = ann[ann["label"].str.fullmatch(r"\d")].copy()
    ann = ann[~ann["label"].isin(ignore_labels)].copy()

    ann = ann.sort_values("dt").reset_index(drop=True)

    return ann


def collapse_duplicate_markers(ann, tolerance_seconds=1):
    """
    Collapse duplicate markers of the same event type that occur very close together.

    This avoids pairing accidental repeated clicks as separate start/end events.
    """
    if ann.empty:
        return ann.copy()

    collapsed = []

    for label, sub in ann.groupby("label"):
        sub = sub.sort_values("dt").copy()

        keep_rows = []
        last_kept_time = None

        for _, row in sub.iterrows():
            if last_kept_time is None:
                keep_rows.append(row)
                last_kept_time = row["dt"]
            else:
                delta_s = (row["dt"] - last_kept_time).total_seconds()

                if delta_s > tolerance_seconds:
                    keep_rows.append(row)
                    last_kept_time = row["dt"]

        collapsed.append(pd.DataFrame(keep_rows))

    if len(collapsed) == 0:
        return ann.iloc[0:0].copy()

    out = pd.concat(collapsed, ignore_index=True)
    out = out.sort_values("dt").reset_index(drop=True)

    return out


def pair_event_markers(ann):
    """
    Pair consecutive markers of the same EEG event type into start/end intervals.

    Assumption:
    For each event type, marker 1 = start, marker 2 = end,
    marker 3 = start, marker 4 = end, etc.
    """
    events = []
    unpaired = []

    event_id = 0

    for label, sub in ann.groupby("label"):
        sub = sub.sort_values("dt").reset_index(drop=True)

        if len(sub) % 2 != 0:
            unpaired.append({
                "event_type": label,
                "n_markers": len(sub),
                "unpaired_marker_time": sub.iloc[-1]["dt"],
            })

        for i in range(0, len(sub) - 1, 2):
            start_time = sub.iloc[i]["dt"]
            end_time = sub.iloc[i + 1]["dt"]

            if end_time <= start_time:
                continue

            events.append({
                "event_id": event_id,
                "event_type": label,
                "start_time": start_time,
                "end_time": end_time,
                "duration_s": (end_time - start_time).total_seconds(),
            })

            event_id += 1

    events_df = pd.DataFrame(events)
    unpaired_df = pd.DataFrame(unpaired)

    if not events_df.empty:
        events_df = events_df.sort_values("start_time").reset_index(drop=True)

    return events_df, unpaired_df


def prepare_variable_timeseries(vars_df, variable_cols, resample_rule="1min"):
    """
    Convert physiological variables into a regular time series.

    This is useful because your Excel table may be recorded every 5 minutes,
    while EEG events can occur between those timestamps.
    """
    ts = vars_df[["dt"] + variable_cols].copy()
    ts = ts.sort_values("dt")
    ts = ts.set_index("dt")

    for col in variable_cols:
        ts[col] = pd.to_numeric(ts[col], errors="coerce")

    ts = ts.resample(resample_rule).mean()
    ts = ts.interpolate(method="time", limit_area="inside")

    return ts


def window_stats(ts, variable, start, end):
    """
    Calculate summary statistics for one variable inside a time window.
    """
    window = ts.loc[(ts.index >= start) & (ts.index <= end), variable].dropna()

    if len(window) == 0:
        return {
            "n_points": 0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "slope_per_min": np.nan,
        }

    if len(window) >= 2:
        x_min = (window.index - window.index[0]).total_seconds() / 60
        y = window.to_numpy(dtype=float)

        try:
            slope = np.polyfit(x_min, y, 1)[0]
        except Exception:
            slope = np.nan
    else:
        slope = np.nan

    return {
        "n_points": len(window),
        "mean": window.mean(),
        "median": window.median(),
        "min": window.min(),
        "max": window.max(),
        "slope_per_min": slope,
    }


def extract_event_variable_features(
    events_df,
    ts,
    variable_cols,
    pre_window_min=5,
    post_window_min=5,
    min_points_per_window=1,
):
    """
    For each EEG event and each physiological variable, extract:
    - baseline/pre-event statistics
    - during-event statistics
    - post-event statistics
    - changes relative to baseline
    """
    rows = []

    for _, event in events_df.iterrows():
        event_id = event["event_id"]
        event_type = event["event_type"]
        start_time = event["start_time"]
        end_time = event["end_time"]

        pre_start = start_time - pd.Timedelta(minutes=pre_window_min)
        pre_end = start_time

        during_start = start_time
        during_end = end_time

        post_start = end_time
        post_end = end_time + pd.Timedelta(minutes=post_window_min)

        for variable in variable_cols:
            pre = window_stats(ts, variable, pre_start, pre_end)
            during = window_stats(ts, variable, during_start, during_end)
            post = window_stats(ts, variable, post_start, post_end)

            if (
                pre["n_points"] < min_points_per_window
                or during["n_points"] < min_points_per_window
            ):
                continue

            rows.append({
                "event_id": event_id,
                "event_type": event_type,
                "variable": variable,
                "start_time": start_time,
                "end_time": end_time,
                "duration_s": event["duration_s"],

                "pre_n": pre["n_points"],
                "during_n": during["n_points"],
                "post_n": post["n_points"],

                "pre_mean": pre["mean"],
                "during_mean": during["mean"],
                "post_mean": post["mean"],

                "pre_median": pre["median"],
                "during_median": during["median"],
                "post_median": post["median"],

                "pre_min": pre["min"],
                "during_min": during["min"],
                "post_min": post["min"],

                "pre_max": pre["max"],
                "during_max": during["max"],
                "post_max": post["max"],

                "pre_slope_per_min": pre["slope_per_min"],
                "during_slope_per_min": during["slope_per_min"],

                "delta_during_mean": during["mean"] - pre["mean"],
                "delta_post_mean": post["mean"] - pre["mean"] if post["n_points"] >= min_points_per_window else np.nan,

                "delta_during_median": during["median"] - pre["median"],
                "delta_post_median": post["median"] - pre["median"] if post["n_points"] >= min_points_per_window else np.nan,
            })

    return pd.DataFrame(rows)


def summarize_event_effects(features_df, min_events_for_stats=3):
    """
    Summarize variable changes by EEG event type and variable.

    Uses Wilcoxon signed-rank test on delta_during_mean and delta_post_mean.
    """
    rows = []

    for (event_type, variable), sub in features_df.groupby(["event_type", "variable"]):
        during_delta = sub["delta_during_mean"].dropna()
        post_delta = sub["delta_post_mean"].dropna()

        during_p = np.nan
        post_p = np.nan

        if len(during_delta) >= min_events_for_stats and not np.allclose(during_delta, 0):
            try:
                during_p = wilcoxon(during_delta).pvalue
            except Exception:
                during_p = np.nan

        if len(post_delta) >= min_events_for_stats and not np.allclose(post_delta, 0):
            try:
                post_p = wilcoxon(post_delta).pvalue
            except Exception:
                post_p = np.nan

        rows.append({
            "event_type": event_type,
            "variable": variable,
            "n_events": sub["event_id"].nunique(),

            "mean_delta_during": during_delta.mean(),
            "median_delta_during": during_delta.median(),
            "std_delta_during": during_delta.std(),

            "mean_delta_post": post_delta.mean(),
            "median_delta_post": post_delta.median(),
            "std_delta_post": post_delta.std(),

            "wilcoxon_p_during": during_p,
            "wilcoxon_p_post": post_p,
        })

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    # Multiple-comparison correction
    if HAS_STATSMODELS:
        for p_col in ["wilcoxon_p_during", "wilcoxon_p_post"]:
            valid = summary[p_col].notna()

            if valid.sum() > 0:
                _, corrected_p, _, _ = multipletests(
                    summary.loc[valid, p_col],
                    method="fdr_bh"
                )
                summary.loc[valid, p_col.replace("p_", "p_fdr_")] = corrected_p
            else:
                summary[p_col.replace("p_", "p_fdr_")] = np.nan
    else:
        summary["wilcoxon_p_fdr_during"] = np.nan
        summary["wilcoxon_p_fdr_post"] = np.nan

    return summary.sort_values(
        ["event_type", "wilcoxon_p_during", "variable"],
        na_position="last"
    ).reset_index(drop=True)


# --------------------------------------------------
# Run analysis
# --------------------------------------------------

ann_numeric = prepare_numeric_annotations(
    comments,
    ignore_labels=EVENTS_TO_IGNORE,
)

ann_numeric = collapse_duplicate_markers(
    ann_numeric,
    tolerance_seconds=DUPLICATE_TOLERANCE_SECONDS,
)

events_df, unpaired_df = pair_event_markers(ann_numeric)

print("Paired EEG events:")
display(events_df)

if not unpaired_df.empty:
    print("Warning: some EEG markers were left unpaired:")
    display(unpaired_df)

ts = prepare_variable_timeseries(
    vars_df,
    variable_cols,
    resample_rule=RESAMPLE_RULE,
)

features_df = extract_event_variable_features(
    events_df=events_df,
    ts=ts,
    variable_cols=variable_cols,
    pre_window_min=PRE_WINDOW_MIN,
    post_window_min=POST_WINDOW_MIN,
    min_points_per_window=MIN_POINTS_PER_WINDOW,
)

summary_df = summarize_event_effects(
    features_df,
    min_events_for_stats=MIN_EVENTS_FOR_STATS,
)

print("Event-variable feature table:")
display(features_df)

print("Summary by EEG event type and variable:")
display(summary_df)

# --------------------------------------------------
# Strongest changes during EEG events
# --------------------------------------------------

strongest_during = summary_df.copy()

strongest_during["abs_median_delta_during"] = strongest_during["median_delta_during"].abs()

strongest_during = strongest_during.sort_values(
    ["event_type", "abs_median_delta_during"],
    ascending=[True, False]
)

display(strongest_during[
    [
        "event_type",
        "variable",
        "n_events",
        "mean_delta_during",
        "median_delta_during",
        "wilcoxon_p_during",
        "wilcoxon_p_fdr_during",
    ]
])


# --------------------------------------------------
# Strongest changes after EEG events
# --------------------------------------------------

strongest_post = summary_df.copy()

strongest_post["abs_median_delta_post"] = strongest_post["median_delta_post"].abs()

strongest_post = strongest_post.sort_values(
    ["event_type", "abs_median_delta_post"],
    ascending=[True, False]
)

display(strongest_post[
    [
        "event_type",
        "variable",
        "n_events",
        "mean_delta_post",
        "median_delta_post",
        "wilcoxon_p_post",
        "wilcoxon_p_fdr_post",
    ]
])

# --------------------------------------------------
# Heatmap of median during-event changes
# --------------------------------------------------

heatmap_data = summary_df.pivot(
    index="variable",
    columns="event_type",
    values="median_delta_during"
)

fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(heatmap_data))))

im = ax.imshow(heatmap_data, aspect="auto")

ax.set_xticks(np.arange(len(heatmap_data.columns)))
ax.set_yticks(np.arange(len(heatmap_data.index)))

ax.set_xticklabels(heatmap_data.columns)
ax.set_yticklabels(heatmap_data.index)

ax.set_xlabel("EEG event type")
ax.set_title("Median change during EEG event vs pre-event baseline")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Median delta during event")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Boxplot of during-event changes for one EEG event type
# --------------------------------------------------

EVENT_TYPE_TO_PLOT = "6"   # change this to "1", "2", "4", etc.

sub = features_df[features_df["event_type"] == EVENT_TYPE_TO_PLOT].copy()

variables_to_plot = (
    sub.groupby("variable")["delta_during_mean"]
    .median()
    .abs()
    .sort_values(ascending=False)
    .head(8)
    .index
    .tolist()
)

box_data = [
    sub.loc[sub["variable"] == var, "delta_during_mean"].dropna()
    for var in variables_to_plot
]

fig, ax = plt.subplots(figsize=(12, 6))

ax.boxplot(box_data, labels=variables_to_plot, vert=True)

ax.axhline(0, linestyle="--", linewidth=1)
ax.set_ylabel("During-event mean - pre-event mean")
ax.set_title(f"Physiological changes during EEG event {EVENT_TYPE_TO_PLOT}")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Event-aligned average trajectory around EEG onset
# --------------------------------------------------

def event_aligned_trajectory(
    events_df,
    ts,
    variable,
    event_type,
    pre_min=10,
    post_min=10,
    step_rule="1min",
):
    """
    Align one variable around onset of a specific EEG event type.
    Returns one row per relative time point.
    """
    selected_events = events_df[events_df["event_type"] == str(event_type)].copy()

    aligned_rows = []

    rel_times = pd.timedelta_range(
        start=-pd.Timedelta(minutes=pre_min),
        end=pd.Timedelta(minutes=post_min),
        freq=step_rule,
    )

    for _, event in selected_events.iterrows():
        onset = event["start_time"]

        for rel_t in rel_times:
            absolute_t = onset + rel_t

            if absolute_t in ts.index:
                value = ts.loc[absolute_t, variable]
            else:
                nearest_idx = ts.index.get_indexer([absolute_t], method="nearest")[0]
                value = ts.iloc[nearest_idx][variable]

            aligned_rows.append({
                "event_id": event["event_id"],
                "event_type": event_type,
                "variable": variable,
                "relative_min": rel_t.total_seconds() / 60,
                "value": value,
            })

    aligned = pd.DataFrame(aligned_rows)

    summary = (
        aligned
        .groupby("relative_min")["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    return aligned, summary


VARIABLE_TO_PLOT = "PAm [mmHg]"   # change this to any variable name
EVENT_TYPE_TO_PLOT = "6"

aligned_raw, aligned_summary = event_aligned_trajectory(
    events_df=events_df,
    ts=ts,
    variable=VARIABLE_TO_PLOT,
    event_type=EVENT_TYPE_TO_PLOT,
    pre_min=10,
    post_min=10,
    step_rule="1min",
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    aligned_summary["relative_min"],
    aligned_summary["mean"],
    marker="o",
    label=f"{VARIABLE_TO_PLOT}, mean"
)

ax.fill_between(
    aligned_summary["relative_min"],
    aligned_summary["mean"] - aligned_summary["sem"],
    aligned_summary["mean"] + aligned_summary["sem"],
    alpha=0.25,
    label="SEM"
)

ax.axvline(0, linestyle="--", linewidth=1.5, label="EEG event onset")
ax.axhline(
    aligned_summary.loc[aligned_summary["relative_min"] < 0, "mean"].mean(),
    linestyle=":",
    linewidth=1,
    label="Pre-event mean"
)

ax.set_xlabel("Time relative to EEG event onset (min)")
ax.set_ylabel(VARIABLE_TO_PLOT)
ax.set_title(f"{VARIABLE_TO_PLOT} around EEG event {EVENT_TYPE_TO_PLOT}")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

summary_df.sort_values("wilcoxon_p_during").head(20)

summary_df.sort_values("wilcoxon_p_fdr_during").head(20)