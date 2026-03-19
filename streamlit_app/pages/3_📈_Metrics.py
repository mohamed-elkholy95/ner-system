"""Metrics page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px

st.title("📈 Model Metrics")
from src.evaluation import compute_ner_metrics, generate_report

# Demo metrics
demo_metrics = {
    "precision": 0.87, "recall": 0.82, "f1": 0.84, "support": 135,
    "per_type": {
        "PER": {"precision": 0.92, "recall": 0.88, "f1": 0.90, "support": 45},
        "ORG": {"precision": 0.85, "recall": 0.80, "f1": 0.82, "support": 32},
        "LOC": {"precision": 0.88, "recall": 0.83, "f1": 0.85, "support": 28},
        "DATE": {"precision": 0.90, "recall": 0.85, "f1": 0.87, "support": 18},
    }
}

col1, col2, col3 = st.columns(3)
col1.metric("Precision", f"{demo_metrics['precision']:.1%}")
col2.metric("Recall", f"{demo_metrics['recall']:.1%}")
col3.metric("F1 Score", f"{demo_metrics['f1']:.1%}")

types = list(demo_metrics["per_type"].keys())
fig = px.bar(x=types, y=[demo_metrics["per_type"][t]["f1"] for t in types],
             labels={"x": "Entity Type", "y": "F1 Score"},
             title="F1 Score by Entity Type",
             color=[demo_metrics["per_type"][t]["f1"] for t in types],
             color_continuous_scale="Blues")
fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown(generate_report(demo_metrics))
