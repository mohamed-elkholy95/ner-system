"""Overview page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px

st.title("📊 Named Entity Recognition — Overview")
st.markdown("Identify and classify named entities (people, organizations, locations, dates) in text.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Entity Types")
    st.markdown("- 👤 **PER** — Person names\n- 🏢 **ORG** — Organizations\n- 📍 **LOC** — Locations\n- 📅 **DATE** — Dates\n- 📎 **MISC** — Miscellaneous")
with col2:
    st.subheader("Models")
    st.markdown("- CRF (Conditional Random Fields)\n- BiLSTM-CRF (PyTorch)\n- BIO tagging scheme\n- Feature engineering")

entity_counts = {"PER": 45, "ORG": 32, "LOC": 28, "DATE": 18, "MISC": 12}
fig = px.bar(x=list(entity_counts.keys()), y=list(entity_counts.values()),
             labels={"x": "Entity Type", "y": "Count"},
             title="Entity Distribution", color=list(entity_counts.values()),
             color_continuous_scale="Blues")
fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white", showlegend=False)
st.plotly_chart(fig, use_container_width=True)
