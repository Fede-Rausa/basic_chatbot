# app.py
import streamlit as st
from huggingface_hub import InferenceClient

st.title("🤖 LLM Chat")


mymodel = "meta-llama/Llama-3.1-8B-Instruct"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = InferenceClient(
        provider="novita",
        api_key=st.secrets["HF_TOKEN"]
    )

    with st.chat_message("assistant"):
        response_stream = client.chat.completions.create(
            model=mymodel,
            messages=st.session_state.messages,
            max_tokens=500,
            stream=True,  # 👈 enable streaming
        )

        # st.write_stream consumes the generator and renders tokens live
        reply = st.write_stream(
            chunk.choices[0].delta.content or ""
            for chunk in response_stream
            if chunk.choices  # 👈 skip chunks with empty choices list
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
