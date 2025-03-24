# Functions for text processing
import re

def extract_questions(generated_text):
    """Extract questions from generated text using regex."""
    questions = re.findall(r"\d+\.\s*(.*?)\n", generated_text)
    if not questions:
        questions = re.findall(r"Q:\s*(.*?)\n", generated_text)
    if not questions:
        questions = [q.strip() for q in generated_text.split("\n") if q.strip()]
    return questions