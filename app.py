import streamlit as st

st.set_page_config(
    page_title="BBAI Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("BBAI Dashboard")
st.write("La webapp è attiva.")

entry_price = st.number_input(
    "Prezzo medio di carico (€)",
    min_value=0.0,
    value=0.0,
    step=0.01,
    format="%.3f"
)

capital = st.number_input(
    "Capitale investito (€)",
    min_value=0.0,
    value=0.0,
    step=50.0,
    format="%.2f"
)

st.write("Prezzo medio inserito:", entry_price)
st.write("Capitale inserito:", capital)
