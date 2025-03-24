import speech_recognition as sr
import pyttsx3
import streamlit as st

def speech_to_text():
    """Capture audio from the microphone and convert it to text in real-time."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        while True:
            try:
                audio = recognizer.listen(source, phrase_time_limit=5)  # Listen for 5 seconds
                text = recognizer.recognize_google(audio)  # Use Google Web Speech API
                yield text  # Yield the recognized text
            except sr.UnknownValueError:
                yield "Could not understand audio."
            except sr.RequestError:
                yield "Speech recognition service unavailable."

def text_to_speech(text):
    """Convert text to speech and play it."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()