# Step 3: Career Pathways
import streamlit as st
from app.utils.llm_utils import suggest_career_pathways

def page3():
    st.title("Career Guidance App")
    st.header("Step 3: Career Pathways")
    
    chat_history = st.session_state.memory.load_memory_variables({})["history"]
    occupation, skills = suggest_career_pathways(
        chat_history, st.session_state.user_input["vector_store"]
    )
    
    st.subheader("Suggested Occupation")
    st.write(occupation)
    
    st.subheader("Skills to Learn")
    st.write(skills)