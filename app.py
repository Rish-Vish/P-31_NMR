# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import re

st.title("NMR Experiment Database")
st.caption("Step 1: Chemical → Step 2: Temperature/Time → Step 3: Cycles → Step 4: Results")

# --- Load data ---
conn = sqlite3.connect("nmr_project.db")
df = pd.read_sql("SELECT * FROM Experiments", conn)
conn.close()

# --- Helpers to parse numeric fields like '265/200', '15x4', '15,30,45', '-' ---
def num_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): 
        return []
    s = str(x)
    # capture integers or decimals
    vals = re.findall(r"\d+(?:\.\d+)?", s)
    return [float(v) for v in vals]

def cycles_norm(x):
    # extract first integer in Cycles; default to 1 if none
    ms = re.findall(r"\d+", str(x))
    return int(ms[0]) if ms else 1

# Precompute numeric arrays for robust filtering
df["Temp_vals"] = df["Temperature (°C)"].apply(num_list)
df["Time_vals"] = df["Time (min)"].apply(num_list)
df["Cycles_norm"] = df["Cycles"].apply(cycles_norm)

# Compute global ranges for defaults
all_temps = [v for vs in df["Temp_vals"] for v in vs]
all_times = [v for vs in df["Time_vals"] for v in vs]
temp_min, temp_max = (min(all_temps), max(all_temps)) if all_temps else (0.0, 0.0)
time_min, time_max = (min(all_times), max(all_times)) if all_times else (0.0, 0.0)

# --- UI: Step 1 — Chemical search ---
chem_query = st.text_input("Step 1 — Search Chemical name or fragment (e.g., MAP, DAP, Phosphoric):", "")

# --- UI: Step 2 — Temperature/Time filters with defaults ---
colA, colB = st.columns(2)

with colA:
    use_temp = st.checkbox("Filter by Temperature (°C)", value=False)
    if use_temp:
        sel_temp = st.slider(
            "Select Temperature range",
            min_value=float(temp_min),
            max_value=float(temp_max if temp_max > temp_min else temp_min + 1.0),
            value=(float(temp_min), float(temp_max if temp_max > temp_min else temp_min + 1.0)),
            step=1.0,
        )
    else:
        sel_temp = (float(temp_min), float(temp_max))  # default range

with colB:
    use_time = st.checkbox("Filter by Time (min)", value=False)
    if use_time:
        sel_time = st.slider(
            "Select Time range",
            min_value=float(time_min),
            max_value=float(time_max if time_max > time_min else time_min + 1.0),
            value=(float(time_min), float(time_max if time_max > time_min else time_min + 1.0)),
            step=1.0,
        )
    else:
        sel_time = (float(time_min), float(time_max))  # default range

# --- UI: Step 3 — Cycles ---
cycles_choice = st.selectbox("Step 3 — Number of cycles", options=["Any", 1, 4], index=0)

# --- Filtering (Step 4) ---
mask = pd.Series(True, index=df.index)

# Chemical search across the 'Chemicals & Reagents' column
if chem_query.strip():
    mask &= df["Chemicals & Reagents"].astype(str).str.contains(chem_query.strip(), case=False, na=False)

# Temperature filter: include rows where ANY parsed temperature falls within chosen range
t_lo, t_hi = sel_temp
if use_temp:
    temp_mask = df["Temp_vals"].apply(lambda vs: any((v >= t_lo) and (v <= t_hi) for v in vs) if len(vs) > 0 else False)
    mask &= temp_mask

# Time filter: include rows where ANY parsed time falls within chosen range
ti_lo, ti_hi = sel_time
if use_time:
    time_mask = df["Time_vals"].apply(lambda vs: any((v >= ti_lo) and (v <= ti_hi) for v in vs) if len(vs) > 0 else False)
    mask &= time_mask

# Cycles filter
if cycles_choice != "Any":
    mask &= (df["Cycles_norm"] == int(cycles_choice))

result = df[mask].copy()

# Clean helper columns from display
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

