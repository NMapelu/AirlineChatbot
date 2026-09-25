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

# --- 2. Load Model & Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
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

# --- 3. Domain filter ---
# Words strongly associated with airline travel.
AIRLINE_KEYWORDS = {
    # Flight-related
    "flight", "flights", "fly", "flying", "airline", "airlines", "airways",
    "plane", "airplane", "aircraft", "departure", "depart", "arrival", "arrive",
    "takeoff", "landing", "layover", "stopover", "nonstop", "direct",
    # Booking / reservation
    "book", "booking", "reserve", "reservation", "itinerary", "ticket", "tickets",
    "fare", "fares", "price", "prices", "cost", "quote",
    # Trip / travel
    "trip", "travel", "journey", "vacation", "holiday", "destination",
    # Airport
    "airport", "terminal", "gate", "boarding", "check-in", "checkin",
    # Baggage
    "bag", "bags", "baggage", "luggage", "suitcase", "carry-on", "carryon",
    "checked", "oversize", "overweight",
    # Seats
    "seat", "seats", "seating", "aisle", "window", "legroom", "upgrade",
    # Changes / cancellations
    "cancel", "cancellation", "change", "reschedule", "rebook", "refund",
    "reimbursement", "compensation",
    # Insurance
    "insurance", "coverage", "cover", "policy", "claim",
    # Passengers
    "passenger", "passengers", "traveler", "traveller",
    # Status / info
    "status", "delay", "delayed", "boarding", "pass", "confirmation",
    # Assistance
    "agent", "human", "representative", "help", "assistance", "support",
    # Airport codes and cities commonly asked about
    "london", "paris", "new york", "tokyo", "dubai", "nairobi", "lagos",
    "amsterdam", "frankfurt", "rome", "madrid", "berlin", "toronto",
    "sydney", "singapore", "hong kong", "mumbai", "delhi", "cairo",
}

def is_airline_related(text: str) -> bool:
    """True if text contains at least one airline-related keyword."""
    tokens = re.findall(r"[a-z0-9\-]+", text.lower())
    return any(tok in AIRLINE_KEYWORDS for tok in tokens)

# --- 4. Chat logic ---
CONFIDENCE_THRESHOLD = 0.35

OFF_TOPIC_RESPONSE = (
    "I'm an airline travel assistant, so I can only help with flights, "
    "bookings, baggage, seats, trip insurance, cancellations, refunds, and "
    "similar travel topics. Could you rephrase your question about your trip?"
)

UNCERTAIN_RESPONSE = (
    "I'm not quite sure what you're asking. Could you rephrase, or tell me "
    "whether it's about booking a flight, baggage, seats, cancellations, "
    "refunds, or trip insurance?"
)

def clean_text(text):
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)

def get_bot_response(user_input):
    # Layer 1: domain check
    if not is_airline_related(user_input):
        return OFF_TOPIC_RESPONSE, "off_topic", 0.0

    # Layer 2: model prediction + confidence
    cleaned = clean_text(user_input)
    probs = model.predict_proba([cleaned])[0]
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    intent = label_map[top_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return UNCERTAIN_RESPONSE, intent, confidence

    return response_map.get(intent, UNCERTAIN_RESPONSE), intent, confidence

# --- 5. Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response_text, intent, confidence = get_bot_response(prompt)

    with st.chat_message("assistant"):
        st.markdown(response_text)
        with st.expander("Debug info"):
            st.write(f"Predicted Intent: `{intent}`")
            st.write(f"Confidence: `{confidence:.2f}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )