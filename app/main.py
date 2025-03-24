# Entry point for the Streamlit app
import streamlit as st
from app.pages.page_1 import page1
from app.pages.page_2 import page2
from app.pages.page_3 import page3

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            display: none;
        }
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    if "page" not in st.session_state:
        st.session_state.page = 1
    
    if st.session_state.page == 1:
        page1()
    elif st.session_state.page == 2:
        page2()
    elif st.session_state.page == 3:
        page3()

if __name__ == "__main__":
    main()