import streamlit as st
from modules.M2_V1_Information_extraction import legal_rag_query

st.title("Legal Sanjeevani ⚖️")

# Input fields
case_id = st.text_input("Enter Case ID")
query = st.text_input("Enter your legal question")

# Button
if st.button("Run"):
    result = legal_rag_query(query, case_id)
    st.write(result)
