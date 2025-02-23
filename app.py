import streamlit as st
import os
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader  # For PDF resume parsing
from langchain_community.vectorstores import FAISS  # For vector storage
from langchain_community.embeddings import OllamaEmbeddings  # For embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text
import re

# Initialize Ollama with Llama3.1
llm = Ollama(model="llama3.1")
embeddings = OllamaEmbeddings(model="llama3.1")  # For creating embeddings

# Streamlit app configuration
st.set_page_config(page_title="Career Guidance App", layout="wide")

# Function to save uploaded file temporarily
def save_uploaded_file(uploaded_file):
    if not os.path.exists("temp"):
        os.makedirs("temp")
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to load and process resume (PDF or text)
def process_resume(resume):
    if resume.type == "application/pdf":
        # Save the uploaded file temporarily
        file_path = save_uploaded_file(resume)
        # Use PyPDFLoader for PDF resumes
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text = " ".join([page.page_content for page in pages])
    elif resume.type == "text/plain":
        # Read text directly from TXT files
        text = resume.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")
    
    # Split text into chunks for vector storage
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Create a vector store from the resume chunks
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Function to generate initial questions using resume data
def generate_initial_questions(user_input, vector_store):
    # Retrieve relevant resume chunks from the vector store
    relevant_chunks = vector_store.similarity_search(user_input, k=3)
    resume_context = " ".join([chunk.page_content for chunk in relevant_chunks])
    
    # Generate questions using the LLM
    prompt = PromptTemplate(
        input_variables=["user_input", "resume_context"],
        template="""
        Based on the following user input and resume context, generate 10 questions for an interview:
        User Input: {user_input}
        Resume Context: {resume_context}
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(user_input=user_input, resume_context=resume_context)

# Function to conduct the interview
def conduct_interview(question, memory):
    conversation = ConversationChain(llm=llm, memory=memory)
    response = conversation.run(question)
    return response

# Function to suggest occupation and skills
def suggest_career_pathways(chat_history, vector_store):
    # Retrieve relevant resume chunks from the vector store
    relevant_chunks = vector_store.similarity_search(chat_history, k=3)
    resume_context = " ".join([chunk.page_content for chunk in relevant_chunks])
    
    # Step 1: Suggest occupation
    occupation_prompt = PromptTemplate(
        input_variables=["chat_history", "resume_context"],
        template="""
        Based on the following chat history and resume context, suggest the most suitable occupation:
        Chat History: {chat_history}
        Resume Context: {resume_context}
        """
    )
    occupation_chain = LLMChain(llm=llm, prompt=occupation_prompt)
    occupation = occupation_chain.run(chat_history=chat_history, resume_context=resume_context)
    
    # Step 2: Suggest skills to learn
    skills_prompt = PromptTemplate(
        input_variables=["occupation"],
        template="""
        Based on the occupation '{occupation}', suggest 5 key skills to learn:
        """
    )
    skills_chain = LLMChain(llm=llm, prompt=skills_prompt)
    skills = skills_chain.run(occupation=occupation)
    
    return occupation, skills

# Function to extract questions from generated text
def extract_questions(generated_text):
    # Use regex to find numbered questions (e.g., "1. What is your experience?")
    questions = re.findall(r"\d+\.\s*(.*?)\n", generated_text)
    
    # If no numbered questions are found, try another pattern (e.g., "Q: What is your experience?")
    if not questions:
        questions = re.findall(r"Q:\s*(.*?)\n", generated_text)
    
    # If still no questions are found, split by newlines and assume each line is a question
    if not questions:
        questions = [q.strip() for q in generated_text.split("\n") if q.strip()]
    
    return questions

# Page 1: Basic Questions and Resume Upload
def page1():
    st.title("Career Guidance App")
    st.header("Step 1: Tell Us About Yourself")
    
    # Basic questions
    name = st.text_input("What is your name?")
    age = st.number_input("How old are you?", min_value=0, max_value=100)
    personality = st.text_area("Describe your personality in a few words:")
    work_experience = st.text_area("Briefly describe your work experience:")
    
    # Resume upload
    resume = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
    
    if st.button("Submit"):
        if name and age and personality and work_experience and resume:
            # Process the resume and create a vector store
            vector_store = process_resume(resume)
            
            # Store user input and vector store in session state
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


# Page 2: Interview Questions with Cross-Questioning
def page2():
    st.title("Career Guidance App")
    st.header("Step 2: Interview")
    
    # Initialize session state variables if they don't exist
    if "questions" not in st.session_state:
        # Generate initial questions using the resume vector store
        questions_text = generate_initial_questions(
            st.session_state.user_input["work_experience"],
            st.session_state.user_input["vector_store"]
        )
        
        # Extract only the questions from the generated text
        st.session_state.questions = extract_questions(questions_text) 
        
        st.session_state.current_question_index = 0
        st.session_state.memory = ConversationBufferMemory()
        st.session_state.cross_question_count = 0  # Track cross-questioning
        st.session_state.conversation_history = []  # Store conversation history for the current question
        st.session_state.waiting_for_followup = False  # Track if we're waiting for a follow-up answer
        st.session_state.followup_questions = []  # Store follow-up questions for the current question
    
    # Display the current question
    if st.session_state.current_question_index < len(st.session_state.questions):
        question = st.session_state.questions[st.session_state.current_question_index]
        
        # Display the conversation history for the current question
        st.write("### Conversation History")
        for entry in st.session_state.conversation_history:
            st.write(f"**{entry['role']}**: {entry['text']}")
        
        # Display the current question or follow-up question
        if not st.session_state.waiting_for_followup:
            st.write(f"**Question {st.session_state.current_question_index + 1}**: {question}")
        else:
            st.write(f"**Follow-up Question {st.session_state.cross_question_count}**: {st.session_state.followup_questions[-1]}")
        
        # Get the user's answer
        user_answer = st.text_input("Your answer:", key=f"answer_{st.session_state.current_question_index}_{st.session_state.cross_question_count}", value="")
        
        # Proceed to cross-questioning or move to the next question
        if user_answer:
            # Save the user's answer to memory
            if not st.session_state.waiting_for_followup:
                st.session_state.memory.save_context({"input": question}, {"output": user_answer})
            else:
                st.session_state.memory.save_context(
                    {"input": st.session_state.followup_questions[-1]}, 
                    {"output": user_answer}
                )
            
            # Add the user's answer to the conversation history
            st.session_state.conversation_history.append({"role": "You", "text": user_answer})
            
            # Cross-questioning logic
            if st.session_state.cross_question_count < 3:
                # Generate a follow-up question based on the user's answer
                cross_question_prompt = PromptTemplate(
                    input_variables=["user_answer"],
                    template="""
                    Based on the following answer, ask a follow-up question:
                    Answer: {user_answer}
                    """
                )
                cross_question_chain = LLMChain(llm=llm, prompt=cross_question_prompt)
                cross_question = cross_question_chain.run(user_answer=user_answer)
                
                # Add the follow-up question to the list of follow-up questions
                st.session_state.followup_questions.append(cross_question)
                
                # Add the follow-up question to the conversation history
                st.session_state.conversation_history.append({"role": "Interviewer", "text": cross_question})
                
                # Set waiting_for_followup to True and increment cross_question_count
                st.session_state.waiting_for_followup = True
                st.session_state.cross_question_count += 1
                
                # Clear the textbox for the next answer
                st.rerun()  # Rerun to refresh the textbox
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
        
# Page 3: Career Pathways (Simplified)
def page3():
    st.title("Career Guidance App")
    st.header("Step 3: Career Pathways")
    
    chat_history = st.session_state.memory.load_memory_variables({})["history"]
    occupation, skills = suggest_career_pathways(
        chat_history, st.session_state.user_input["vector_store"]
    )
    
    # Display suggested occupation
    st.subheader("Suggested Occupation")
    st.write(occupation)
    
    # Display suggested skills
    st.subheader("Skills to Learn")
    st.write(skills)

# Main app logic
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