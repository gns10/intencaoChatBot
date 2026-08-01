import streamlit as st

from src.predict import predict_intent
from src.response_service import get_response

st.set_page_config(
    page_title="Intent Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Intent Chatbot")
st.caption("Classificação de intenções com BiLSTM + CLINC150")

# Estado da conversa
if "messages" not in st.session_state:
    st.session_state.messages = []

if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50

with st.sidebar:
    st.header("Configurações")
    threshold = st.slider(
        "Confiança mínima",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="Abaixo desse valor, a intenção é tratada como incerta.",
    )
    st.session_state.threshold = threshold

    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.write("**Modelo:** feelings.keras")
    st.write("**Classes:** 150 intenções")
    st.write(f"**Threshold:** {threshold:.0%}")

# Histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("metadata"):
            meta = message["metadata"]
            st.caption(
                f"Intenção: `{meta['intent']}` · "
                f"Confiança: `{meta['confidence']:.2%}`"
            )

            if meta.get("url"):
                st.link_button("Abrir link", meta["url"])

# Entrada
prompt = st.chat_input("Digite sua mensagem...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando sua mensagem..."):
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
                        "Não tenho confiança suficiente para identificar "
                        "sua intenção. Pode reformular a pergunta?"
                    )

                st.markdown(answer)

                if result["accepted"] and response_data.get("url"):
                    st.link_button(
                        "Acessar conteúdo",
                        response_data["url"],
                    )

                st.caption(
                    f"Intenção: `{intent}` · "
                    f"Confiança: `{confidence:.2%}`"
                )

                metadata = {
                    "intent": intent,
                    "confidence": confidence,
                    "url": response_data.get("url"),
                }

            except Exception as e:
                answer = (
                    "Ocorreu um erro ao processar a mensagem. "
                    "Verifique se os arquivos do modelo estão disponíveis."
                )
                st.error(answer)
                st.exception(e)
                metadata = None

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "metadata": metadata,
    })
