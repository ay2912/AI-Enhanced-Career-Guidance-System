import streamlit as st
from app.utils.llm_utils import generate_initial_questions
from app.utils.text_utils import extract_questions
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from app.utils.llm_utils import llm
from app.utils.speech_utils import speech_to_text, text_to_speech
from app.config import EXAMPLE_QUESTIONS_PATH

def page2():
    st.title("Career Guidance App")
    st.header("Step 2: Interview")
    
    if "questions" not in st.session_state:
        # Load example questions from the file
        with open(EXAMPLE_QUESTIONS_PATH, "r") as f:
            example_questions = f.read()
        
        # Generate initial questions
        questions_text = generate_initial_questions(
            user_input=st.session_state.user_input["work_experience"],
            vector_store=st.session_state.user_input["vector_store"],
            example_questions=example_questions
        )
        
        # Extract questions from the generated text
        st.session_state.questions = extract_questions(questions_text)
        st.session_state.current_question_index = 0
        st.session_state.memory = ConversationBufferMemory()
        st.session_state.cross_question_count = 0
        st.session_state.conversation_history = []
        st.session_state.waiting_for_followup = False
        st.session_state.followup_questions = []
    
    # Display the current question
    if st.session_state.current_question_index < len(st.session_state.questions):
        question = st.session_state.questions[st.session_state.current_question_index]
        
        # Display the conversation history
        for entry in st.session_state.conversation_history:
            st.write(f"**{entry['role']}**: {entry['text']}")
        
        # Display the current question or follow-up question
        if not st.session_state.waiting_for_followup:
            st.write(f"**Question {st.session_state.current_question_index + 1}**: {question}")
            text_to_speech(question)  # Speak the question
        else:
            st.write(f"**Follow-up Question {st.session_state.cross_question_count}**: {st.session_state.followup_questions[-1]}")
            text_to_speech(st.session_state.followup_questions[-1])  # Speak the follow-up question
        
        # Real-time speech-to-text input
        st.write("### Your Answer")
        if st.button("Start Recording"):
            st.session_state.recording = True
        
        if st.session_state.get("recording", False):
            # Display recognized text in real-time
            recognized_text = st.empty()  # Placeholder for dynamic text
            for text in speech_to_text():
                recognized_text.write(f"**You said**: {text}")
                
                # Stop recording if the user says "stop"
                if "stop" in text.lower():
                    st.session_state.recording = False
                    recognized_text.write("Recording stopped.")
                    break
                
                # Save the recognized text to memory
                if not st.session_state.waiting_for_followup:
                    st.session_state.memory.save_context({"input": question}, {"output": text})
                else:
                    st.session_state.memory.save_context(
                        {"input": st.session_state.followup_questions[-1]}, 
                        {"output": text}
                    )
                
                # Add the recognized text to the conversation history
                st.session_state.conversation_history.append({"role": "You", "text": text})
                
                # Cross-questioning logic
                if st.session_state.cross_question_count < 3:
                    # Generate a follow-up question based on the recognized text
                    cross_question_prompt = PromptTemplate(
                        input_variables=["user_answer"],
                        template="""
                        Based on the following answer, ask a follow-up question:
                        Answer: {user_answer}
                        """
                    )
                    cross_question_chain = LLMChain(llm=llm, prompt=cross_question_prompt)
                    cross_question = cross_question_chain.run(user_answer=text)
                    
                    # Add the follow-up question to the list of follow-up questions
                    st.session_state.followup_questions.append(cross_question)
                    
                    # Add the follow-up question to the conversation history
                    st.session_state.conversation_history.append({"role": "Interviewer", "text": cross_question})
                    
                    # Set waiting_for_followup to True and increment cross_question_count
                    st.session_state.waiting_for_followup = True
                    st.session_state.cross_question_count += 1
                    
                    # Rerun the app to refresh the UI
                    st.rerun()
                else:
                    # Reset cross-questioning counter and move to the next question
                    st.session_state.cross_question_count = 0
                    st.session_state.waiting_for_followup = False
                    st.session_state.followup_questions = []  # Clear follow-up questions for the next question
                    
                    # Prompt before moving to the next question
                    st.write("**Interviewer**: Okay, moving to the next question.")
                    st.session_state.conversation_history = []  # Clear chat history for the next question
                    
                    # Move to the next question
                    st.session_state.current_question_index += 1
                    
                    # If all questions are answered, move to Page 3
                    if st.session_state.current_question_index >= len(st.session_state.questions):
                        st.session_state.page = 3
                        st.rerun()
                    else:
                        # Rerun the app to display the next question
                        st.rerun()
    else:
        st.write("All questions have been answered. Moving to the next step...")
        st.session_state.page = 3
        st.rerun()