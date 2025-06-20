import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch

# Cache the model to avoid reloading
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Cache the QnA data and embeddings
@st.cache_data
def load_qna_data():
    with open("qna.json", "r") as f:
        qna_data = json.load(f)
    
    questions = [entry['question'] for entry in qna_data]
    answers = [entry['answer'] for entry in qna_data]
    return questions, answers

@st.cache_data
def generate_embeddings(questions):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)

def get_best_answer(user_query, questions, answers, question_embeddings):
    model = load_model()
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    best_idx = scores.argmax().item()
    confidence = scores[best_idx].item()
    return answers[best_idx], confidence

# Streamlit UI
st.set_page_config(page_title="Staff Chatbot", page_icon="🤖")

st.title("🤖 Staff Chatbot")
st.write("Ask me anything from our knowledge base!")

# Load data
questions, answers = load_qna_data()
question_embeddings = generate_embeddings(questions)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    response, confidence = get_best_answer(prompt, questions, answers, question_embeddings)
    
    # Add confidence indicator if low
    if confidence < 0.5:
        response = f"⚠️ I'm not very confident about this answer (confidence: {confidence:.2f}):\n\n{response}"
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This chatbot uses semantic search to find the best answers from our knowledge base.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()