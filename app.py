# run: streamlit run app.py
import streamlit as st
from llm import llm
from pipeline import audit

st.title("LLM Hallucination Auditor")
query = st.text_input("Question")
draft = st.text_area("LLM answer (leave empty to let the LLM answer)")
use_search = st.checkbox("Use web search", value=True)

if st.button("Audit") and query:
    with st.spinner("Auditing..."):
        if not draft.strip():
            draft = llm(f"Answer clearly.\nQuestion: {query}")
        out = audit(query, draft, use_search=use_search)
    color = {"VERIFIED": "green", "CORRECTED": "orange",
             "FLAGGED": "orange", "BLOCKED": "red"}[out["action"]]
    st.markdown(f"### :{color}[{out['action']}]  |  R = {out['R']}  |  {out['latency']} s")
    st.subheader("Original answer")
    st.write(draft)
    st.subheader("Final answer")
    st.write(out["final_text"] if out["final_text"] else "Response blocked: too many unsupported claims.")
    st.subheader("Claim-by-claim")
    for c in out["claims"]:
        icon = {"SUP": "✅", "CON": "❌", "NEI": "❓"}[c["verdict"]]
        st.markdown(f"{icon} **{c['claim']}**  \n_{c['reason']}_")
        for e in c["evidence"]:
            st.markdown(f"- [{e['title']}]({e['url']})")
