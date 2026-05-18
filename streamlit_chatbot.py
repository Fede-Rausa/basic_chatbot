# app.py
import streamlit as st
from huggingface_hub import InferenceClient
import re
import numpy as np

load_dotenv()


st.title("LLM Chat",
         help="""
This is a minimal example that shows how making a chatbot for free is possible.

I had this struggle for a long time:

 I don't have a GPU, but I wanted to learn and experiment with LLMs and AI agents.
Furthermore, if I want to host it on a website you need a server that gives me enough resources.

So I exploited two famous free services:
- streamlit, that hosts for free python apps
- HuggingFace, that provides free access to various models through their inference API

The inference apis of Claude and OpenAI are often paid, and limited to some models.

Fortunately, there are free alternatives available on HuggingFace.
The ingredients You need are these:
- Streamlit
- HuggingFace inference API token (is free, to make one you need a HuggingFace account)
- the name of an inference provider active on HuggingFace
- the name of a text generation model that is supported by the provider you have chosen

There are obviously limitations:
- an HuggingFace token cannot be used for free to make hundreds of requests per hour (it can vary depending by the provider and the model size). However, they are more than enough to solve small tasks.
- it is convenient to use only small models (from 2B to 8B parameters there is the best trade-off in my opinion between thinking performance and computational cost)

The code of this demo is published on Github.
         """)

st.write('Minimal example of how to make a chatbot for free, using streamlit and HuggingFace.')


if 'setup' not in st.session_state:
    models_dict = {
    'novita' : {'models':['meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct', 'Sao10K/L3-8B-Stheno-v3.2', 'NousResearch/Hermes-2-Pro-Llama-3-8B'], 'nparams': ['8B', '1B', '8B', '8B']},
    'together': {'models':['Qwen/Qwen2.5-7B-Instruct', 'EssentialAI/rnj-1-instruct'], 'nparams': ['8B', '8B']},
    'cerebras': {'models':['meta-llama/Llama-3.1-8B-Instruct'], 'nparams': ['8B']},
    'nscale': {'models':['Qwen/Qwen3-8B', 'Qwen/Qwen3-4B-Instruct-2507', 
                         'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen/Qwen2.5-Coder-3B-Instruct', 'Qwen/Qwen2.5-Coder-7B-Instruct',
                         'Qwen/Qwen3-4B-Thinking-2507', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'], 
               'nparams': ['8B', '4B', '1.5B', '3B', '7B', '4B', '7B']},
    #'featherless-ai': {'models':['Qwen/Qwen3-1.7B', 'microsoft/phi-2', 'google/gemma-2b-it', 'google/gemma-7b', 'unsloth/gemma-3-1b-pt'], 
    #                   'nparams': ['1.7B', '3B', '2B', '9B', '1B']},
    'featherless-ai' : {'models':['ishaanxgupta/gemma-2-2bit-quantised'], 'nparams': ['3B']},
    'hf-inference' : {'models':['katanemo/Arch-Router-1.5B'], 'nparams': ['1.5B']},
    'publicai' : {'models':['allenai/Olmo-3-7B-Instruct', 'swiss-ai/Apertus-8B-Instruct-2509'], 'nparams': ['0.5B', '8B']},
    'cohere': {'models':['CohereLabs/c4ai-command-r7b-12-2024', 'CohereLabs/tiny-aya-fire', 'CohereLabs/tiny-aya-water', 'CohereLabs/tiny-aya-global', 'CohereLabs/tiny-aya-earth'],
                'nparams': ['7B', '3B', '3B', '3B', '3B']}
    }

    all_models = []
    all_providers = []
    for provider, info in models_dict.items():
        all_models.extend(info['models'])
        all_providers.extend([provider] * len(info['models']))

    st.session_state.all_models = all_models
    st.session_state.all_providers = all_providers
    st.session_state.models_dict = models_dict

    nop = np.array([1, 3, 7, 7, 13, 13, 30, 30,  34, 34, 70, 70])
    enjoule = np.array([5, 15, 30, 60, 60, 120, 150, 300, 150, 300, 300, 700])

    x = nop
    y = enjoule

    beta = np.cov(x, y)[0, 1] / np.var(x)

    inter = np.mean(y) - beta * np.mean(x)

    def predict_mj(nop):
        return max(beta * nop + inter, 0.01)

    st.session_state.predict_mj = predict_mj
    st.session_state.setup = True



mymodel = st.selectbox('Select model', options=sorted(st.session_state.all_models), index=4)

myprovider = st.session_state.all_providers[st.session_state.all_models.index(mymodel)]

nop = st.session_state.models_dict[myprovider]['nparams'][st.session_state.models_dict[myprovider]['models'].index(mymodel)]

st.markdown(f"""
            - **Model:** {mymodel.split('/')[1]}
            - **Inference provider:** {myprovider}
            - **Model author:** {mymodel.split('/')[0]}
            - **Model parameters:** {nop}
            - **Estimated mJ energy consumption per token:** {st.session_state.predict_mj(float(nop.replace('B', ''))):.2f} mJ
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'old_model' not in st.session_state:
    st.session_state.old_model = mymodel
else:
    if st.session_state.old_model != mymodel:
        st.session_state.messages = []
        st.session_state.old_model = mymodel

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = InferenceClient(
        provider=myprovider,
        api_key= st.secrets["HF_TOKEN"]
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
