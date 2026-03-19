"""Entity extraction page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from src.data_generator import bio_to_entities, generate_ner_data, tokens_to_text
from src.ner_model import CRFTagger, sent2features

st.title("🏷️ Extract Entities")

text_input = st.text_area("Enter text", "John Smith visited Google headquarters in New York last Monday.", height=100)

if st.button("Extract Entities", type="primary"):
    tokens = text_input.split()
    # Use mock tagging for demo
    from src.ner_model import CRFTagger
    tagger = CRFTagger()
    tags = tagger.predict([tokens])[0]
    entities = bio_to_entities(tokens, tags)

    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tokens", len(tokens))
        st.metric("Entities Found", len(entities))
    with col2:
        st.metric("Entity Types", len(set(e["type"] for e in entities)))

    if entities:
        import pandas as pd
        df = pd.DataFrame(entities)
        st.dataframe(df, use_container_width=True)

    # Highlighted text
    highlighted = text_input
    st.subheader("Annotated Text")
    for e in entities:
        st.markdown(f"- **{e['text']}** → `{e['type']}`")
