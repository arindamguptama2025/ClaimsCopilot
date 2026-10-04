import pathlib
import pandas as pd
import streamlit as st
from src.rag import ingest, answer
from src.sql_assistant import nl_to_sql
from src.audit import recent

st.set_page_config(page_title="ClaimsCopilot", layout="wide")
st.title("🛡️ ClaimsCopilot")
tab1, tab2, tab3 = st.tabs(["📄 Policy Q&A", "📊 Claims Data", "🧾 Audit Log"])

with tab1:
    files = st.file_uploader("Upload policy documents", type=["pdf", "txt"],
                             accept_multiple_files=True)
    for f in files or []:
        dest = pathlib.Path("data/policies") / f.name
        dest.write_bytes(f.getbuffer())
        st.caption(f"Indexed {f.name}: {ingest(dest)} chunks")
    q = st.text_input("Ask a question", key="rag_q",
                      placeholder="Is water damage from a burst pipe covered?")
    if q:
        with st.spinner("Searching policies..."):
            ans, hits = answer(q)
        st.markdown(ans)
        with st.expander("Sources"):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**[{i}] {h['meta']['source']} (p.{h['meta']['page']})**")
                st.text(h["text"])

with tab2:
    q2 = st.text_input("Ask about claims data", key="sql_q",
                       placeholder="Show open auto claims over 10000 in Lombardy")
    if q2:
        with st.spinner("Generating query..."):
            sql, df, err = nl_to_sql(q2)
        if sql:
            st.code(sql, language="sql")
        if err:
            st.error(err)
        elif df is not None:
            st.dataframe(df)
            if (df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1])
                    and not pd.api.types.is_numeric_dtype(df.iloc[:, 0])):
                st.bar_chart(df.set_index(df.columns[0]))

with tab3:
    st.caption("Every LLM call: masked prompt, answer, sources, PII types found, refusals.")
    st.dataframe(recent(50))