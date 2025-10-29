import streamlit as st

with st.sidebar:
    st.markdown("### Credits")
    st.markdown("[Manav Seth — GitHub](https://github.com/orgs/P31-NMR-2025/repositories)")
    st.markdown("[Rishik Vishwakarma — GitHub](https://github.com/rish-vish)")


# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import re
from pathlib import Path

st.set_page_config(page_title="NMR Experiment Database", layout="wide")

# --- Bootstrap: create/refresh DB table from Excel (cloud-safe, small data) ---
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "nmr_project.db"
XLSX    = ROOT / "NMR_Experiments.xlsx"
TABLE   = "Experiments"

def build_db_from_excel():
    if not XLSX.exists():
        raise FileNotFoundError(f"Excel file not found: {XLSX}")
    df_x = pd.read_excel(XLSX)  # requires openpyxl
    with sqlite3.connect(DB_PATH) as conn:
        df_x.to_sql(TABLE, conn, if_exists="replace", index=False)

# Always rebuild on start for deterministic first run (tiny dataset)
build_db_from_excel()

# Single source of truth for df
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)

st.title("NMR Experiment Database")
st.caption("Step 1: Chemical → Step 2: Temperature/Time → Step 3: Cycles → Step 4: Results")

# --- Helpers to parse numeric fields like '265/200', '15x4', '15,30,45', '-' ---
def num_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x)
    vals = re.findall(r"\d+(?:\.\d+)?", s)  # extract numbers robustly
    return [float(v) for v in vals]

def cycles_norm(x):
    ms = re.findall(r"\d+", str(x))
    return int(ms[0]) if ms else 1

# Precompute numeric arrays for robust filtering
df["Temp_vals"]  = df["Temperature (°C)"].apply(num_list)
df["Time_vals"]  = df["Time (min)"].apply(num_list)
df["Cycles_norm"] = df["Cycles"].apply(cycles_norm)

# Compute global ranges for defaults
all_temps = [v for vs in df["Temp_vals"] for v in vs]
all_times = [v for vs in df["Time_vals"] for v in vs]
temp_min, temp_max = (min(all_temps), max(all_temps)) if all_temps else (0.0, 0.0)
time_min, time_max = (min(all_times), max(all_times)) if all_times else (0.0, 0.0)

# --- UI: Step 1 — Chemical search ---
chem_query = st.text_input("Step 1 — Search Chemical (e.g., MAP, DAP, Phosphoric):", "")

# --- UI: Step 2 — Temperature/Time filters with defaults ---
colA, colB = st.columns(2)
with colA:
    use_temp = st.checkbox("Filter by Temperature (°C)", value=False)
    if use_temp:
        sel_temp = st.slider(
            "Temperature range",
            min_value=float(temp_min),
            max_value=float(temp_max if temp_max > temp_min else temp_min + 1.0),
            value=(float(temp_min), float(temp_max if temp_max > temp_min else temp_min + 1.0)),
            step=1.0,
        )
    else:
        sel_temp = (float(temp_min), float(temp_max))
with colB:
    use_time = st.checkbox("Filter by Time (min)", value=False)
    if use_time:
        sel_time = st.slider(
            "Time range",
            min_value=float(time_min),
            max_value=float(time_max if time_max > time_min else time_min + 1.0),
            value=(float(time_min), float(time_max if time_max > time_min else time_min + 1.0)),
            step=1.0,
        )
    else:
        sel_time = (float(time_min), float(time_max))

# --- UI: Step 3 — Cycles ---
cycles_choice = st.selectbox("Step 3 — Number of cycles", options=["Any", 1, 4], index=0)

# --- Filtering (Step 4) ---
mask = pd.Series(True, index=df.index)

if chem_query.strip():
    mask &= df["Chemicals & Reagents"].astype(str).str.contains(chem_query.strip(), case=False, na=False)

t_lo, t_hi = sel_temp
if use_temp:
    temp_mask = df["Temp_vals"].apply(lambda vs: any((v >= t_lo) and (v <= t_hi) for v in vs) if vs else False)
    mask &= temp_mask

ti_lo, ti_hi = sel_time
if use_time:
    time_mask = df["Time_vals"].apply(lambda vs: any((v >= ti_lo) and (v <= ti_hi) for v in vs) if vs else False)
    mask &= time_mask

if cycles_choice != "Any":
    mask &= (df["Cycles_norm"] == int(cycles_choice))

result = df[mask].copy()

# Hide helper columns
display_cols = [c for c in df.columns if c not in ["Temp_vals", "Time_vals", "Cycles_norm"]]

st.markdown("#### Results")
st.write(f"Matches: {len(result)}")
st.dataframe(result[display_cols], use_container_width=True)

st.download_button(
    "Download results (CSV)",
    result[display_cols].to_csv(index=False),
    file_name="nmr_filtered_results.csv",
    mime="text/csv",
)
