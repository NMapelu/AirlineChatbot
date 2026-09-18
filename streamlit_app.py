import streamlit as st
import joblib
import json
import numpy as np
import re
import os

# --- 1. Configuration ---
st.set_page_config(page_title="Airline Chatbot", page_icon="✈️")
st.title("✈️ Airline Chatbot")
st.caption("Ask about flights, baggage, seats, or bookings.")

# --- 2. Load Model & Artifacts (from the repo, not Drive) ---
# IMPORTANT: You must copy your model and JSON files into the repo.
# Create a folder named "artifacts" in the root of your repo and place:
#   - model.joblib
#   - label_map.json
#   - response_map.json
# inside it.

@st.cache_resource
def load_artifacts():
    try:
        # Paths relative to the repo root
        base_path = "artifacts"
        model = joblib.load(os.path.join(base_path, "model.joblib"))
        
        with open(os.path.join(base_path, "label_map.json")) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
            
        with open(os.path.join(base_path, "response_map.json")) as f:
            response_map = json.load(f)
            
        return model, label_map, response_map
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

model, label_map, response_map = load_artifacts()

# --- 3. Chat Logic ---
CONFIDENCE_THRESHOLD = 0.35
FALLBACK = "I'm not sure I understood. Could you rephrase, or ask about flights, baggage, seats, or bookings?"

def clean_text(text):
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)

def get_bot_response(user_input):
    cleaned = clean_text(user_input)
    probs = model.predict_proba([cleaned])[0]
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    intent = label_map[top_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return FALLBACK, intent, confidence
    else:
        reply = response_map.get(intent, FALLBACK)
        return reply, intent, confidence

# --- 4. Chat UI ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    response_text, intent, confidence = get_bot_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_text)
        # Optional: Show confidence in an expander for debugging
        with st.expander("Debug info"):
            st.write(f"Predicted Intent: `{intent}`")
            st.write(f"Confidence: `{confidence:.2f}`")
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})