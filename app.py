import streamlit as st

from src.predict import predict_intent
from src.response_service import get_response

st.set_page_config(
    page_title="Intent Chatbot (EN)",
    page_icon="🤖",
    layout="centered",
)

st.title("Intent Chatbot")
st.caption("Intent classification with BiLSTM + CLINC150")

# Estado da conversa
if "messages" not in st.session_state:
    st.session_state.messages = []

if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50

with st.sidebar:
    st.header("Configurações")
    threshold = st.slider(
        "Minimum confidence",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="Below this value, the intent is treated as uncertain.",
    )
    st.session_state.threshold = threshold

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.write("**Model:** feelings.keras")
    st.write("**Classes:** 150 intents")
    st.write(f"**Threshold:** {threshold:.0%}")

# Histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("metadata"):
            meta = message["metadata"]
            st.caption(
                f"Intent: `{meta['intent']}` · "
                f"Confidence: `{meta['confidence']:.2%}`"
            )

            if meta.get("url"):
                st.link_button("Open link", meta["url"])

# Entrada
prompt = st.chat_input("Enter your message in English...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your message..."):
            try:
                result = predict_intent(
                    prompt,
                    threshold=st.session_state.threshold,
                )

                intent = result["intent"]
                confidence = result["confidence"]
                response_data = get_response(intent)

                if result["accepted"]:
                    answer = response_data["response"]
                else:
                    answer = (
                        "I don't have enough confidence to identify "
                        "your intent. Could you rephrase the question?"
                    )

                st.markdown(answer)

                if result["accepted"] and response_data.get("url"):
                    st.link_button(
                        "Open content",
                        response_data["url"],
                    )

                st.caption(
                    f"Intent: `{intent}` · "
                    f"Confidence: `{confidence:.2%}`"
                )

                metadata = {
                    "intent": intent,
                    "confidence": confidence,
                    "url": response_data.get("url"),
                }

            except Exception as e:
                answer = (
                    "An error occurred while processing the message. "
                    "Please check if the model files are available."
                )
                st.error(answer)
                st.exception(e)
                metadata = None

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "metadata": metadata,
    })
