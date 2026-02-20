"""
Export — Download study results as CSV or JSON.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st

from models import init_db
from services import patient_service, study_service, export_service

init_db()

st.set_page_config(page_title="Export", page_icon="📤", layout="wide")
st.title("Export Study Data")

# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

patients = patient_service.list_patients()
if not patients:
    st.info("No patients found.")
    st.stop()

patient_names = {p.id: p.name for p in patients}
selected_pid = st.selectbox(
    "Select Patient",
    options=[p.id for p in patients],
    format_func=lambda pid: patient_names[pid],
)

studies = study_service.list_studies(selected_pid)
if not studies:
    st.info("No studies for this patient.")
    st.stop()

study_labels = {s.id: f"{s.source_file or s.source} — {s.status}" for s in studies}
selected_sid = st.selectbox(
    "Select Study",
    options=[s.id for s in studies],
    format_func=lambda sid: study_labels[sid],
)

st.divider()

# ---------------------------------------------------------------------------
# Export options
# ---------------------------------------------------------------------------

st.subheader("Export Format")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### CSV")
    st.caption("Feature summaries — compatible with MATLAB, Python, Excel")

    feature_filter = st.selectbox(
        "Filter by feature (optional)",
        options=["All features", "mo_q", "mo_p", "mo_count", "mo_dom_freq"],
    )

    if st.button("Download CSV", use_container_width=True):
        fk = None if feature_filter == "All features" else None
        # For prefix filtering, get all and let the export handle it
        csv_data = export_service.export_csv(selected_sid, feature_key=fk)
        if feature_filter != "All features":
            # Filter CSV rows by prefix
            lines = csv_data.strip().split("\n")
            header = lines[0]
            filtered = [header] + [
                line for line in lines[1:]
                if feature_filter in line
            ]
            csv_data = "\n".join(filtered)

        st.download_button(
            label="Save CSV File",
            data=csv_data,
            file_name=f"study_{selected_sid[:8]}_features.csv",
            mime="text/csv",
        )

with col2:
    st.markdown("### JSON")
    st.caption("Full feature data with metadata — compatible with Python, external tools")

    if st.button("Download JSON", use_container_width=True):
        json_data = export_service.export_json(selected_sid)
        st.download_button(
            label="Save JSON File",
            data=json_data,
            file_name=f"study_{selected_sid[:8]}_full.json",
            mime="application/json",
        )

# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Data Preview")

records = study_service.get_feature_records(selected_sid)
if records:
    preview_data = []
    for r in records[:100]:  # Show first 100 records
        if r.feature_key == "mo_window_detail":
            continue
        preview_data.append({
            "Time (s)": r.timestamp,
            "Feature": r.feature_key,
            "Value": r.feature_value,
        })
    if preview_data:
        st.dataframe(preview_data, use_container_width=True)
        if len(records) > 100:
            st.caption(f"Showing 100 of {len(records)} records. Download full data above.")
else:
    st.info("No data to preview.")
