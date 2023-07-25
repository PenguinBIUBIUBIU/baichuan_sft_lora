import json
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

import sys
if len(sys.argv) >= 4:
    MODEL_NAME_OR_PATH = sys.argv[1]
    LORA_WEIGHTS_PATH = sys.argv[3]
else:
    print("Usage: python chat__web_demo.py [MODEL_NAME_OR_PATH] [LORA_WEIGHTS_PATH]")
    sys.exit(1)

st.set_page_config(
    page_title="baichuan-7B-int4 演示",
    page_icon=":robot:",
    layout='wide'
)
st.title("baichuan-7B-int4")

max_new_tokens = st.sidebar.slider(
    'max_length', 0, 32768, 1024, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.8, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.8, step=0.01
)

@st.cache_resource
def init_model():
    ###加载量化模型
    device_map = {"": 0}
    print(f"{MODEL_NAME_OR_PATH}:加载模型")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH,
                                                 trust_remote_code=True,
                                                 quantization_config=BitsAndBytesConfig(
                                                     load_in_4bit=True,
                                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                                     bnb_4bit_use_double_quant=True,
                                                     bnb_4bit_quant_type='nf4'
                                                 ),
                                                 device_map=device_map)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=2.0,
        max_new_tokens=max_new_tokens,  # max_length=max_new_tokens+input_sequence
    )
    model.generation_config = generation_config

    ###组装lora
    device = "cuda:0"
    print(f"{LORA_WEIGHTS_PATH}:PEFT加载模型")
    model_lora = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS_PATH
    ).to(device)
    model_lora = model_lora.eval()
    return tokenizer, model_lora

tokenizer, model = init_model()

def clear_chat_history():
    del st.session_state.messages

def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是百川大模型，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages

def main():
    model, tokenizer = init_model()
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)

if __name__ == "__main__":
    main()