# Step 1: Basic Questions and Resume Upload
import streamlit as st
from app.utils.file_utils import save_uploaded_file
from app.utils.llm_utils import process_resume

def page1():
    st.title("Career Guidance App")
    st.header("Step 1: Tell Us About Yourself")
    
    name = st.text_input("What is your name?")
    age = st.number_input("How old are you?", min_value=0, max_value=100)
    personality = st.text_area("Describe your personality in a few words:")
    work_experience = st.text_area("Briefly describe your work experience:")
    
    resume = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
    
    if st.button("Submit"):
        if name and age and personality and work_experience and resume:
            vector_store = process_resume(resume)
            st.session_state.user_input = {
                "name": name,
                "age": age,
                "personality": personality,
                "work_experience": work_experience,
                "vector_store": vector_store
            }
            st.session_state.page = 2
            st.rerun()
        else:
            st.error("Please fill out all fields and upload a resume.")