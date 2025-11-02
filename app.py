import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

def import_peft():
    try:
        from peft import PeftModel
        return PeftModel
    except Exception:
        return None

PeftModel = import_peft()

st.set_page_config(page_title="Model UI", layout="centered", initial_sidebar_state="auto")
st.title("Model Playground")
st.sidebar.header("Model configuration")
st.sidebar.write("- merged model folder (full weights + tokenizer) OR")
st.sidebar.write("- base model (HF id or local folder) plus a LoRA adapter folder")
base_model_path = st.sidebar.text_input("Base model path (local folder or HF id)", value="gpt2")
lora_adapter_path = st.sidebar.text_input("LoRA adapter path (folder containing adapter_model.safetensors)", value="./kitchen_gpt_lora_epoch2")
use_gpu_hint = "GPU available" if torch.cuda.is_available() else "GPU not available (CPU inference)"
st.sidebar.markdown(f"**Runtime:** {use_gpu_hint}")
if st.sidebar.button("Load model"):
    st.session_state._load_request = True

@st.cache_resource(show_spinner=False)
def load_model_cached(base_path: str, lora_path: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = None
    for candidate in (base_path, lora_path, "gpt2"):
        if not candidate:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True)
            break
        except Exception:
            pass
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_path, device_map="auto" if device=="cuda" else None)
    if device == "cuda":
        model.to("cuda")
    if lora_path:
        if PeftModel is None:
            raise RuntimeError("PEFT not installed in the environment. Add 'peft' to requirements.")
        model = PeftModel.from_pretrained(model, lora_path, device_map="auto" if device=="cuda" else None)
        if device == "cuda":
            model.to("cuda")
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}

if "_load_request" in st.session_state or (os.path.exists("./final_model") and "model_state" not in st.session_state):
    st.info("Loading model (this can take a minute)...")
    try:
        res = load_model_cached(base_model_path, lora_adapter_path or None)
        st.session_state.model_state = res
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        if "model_state" in st.session_state:
            del st.session_state["model_state"]

st.subheader("Input (use the same format your model expects)")
example_prompt = "TITLE: Cheesy Garlic Bread\nINGREDIENTS: bread; garlic; butter; salt\nDIRECTIONS:"
prompt = st.text_area("Prompt", value=example_prompt, height=160, help="Enter the prompt exactly as during training.")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    max_new_tokens = st.number_input("Max new tokens", min_value=1, max_value=2000, value=200, step=10)
with col2:
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
with col3:
    top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
col4, col5 = st.columns([1,1])
with col4:
    top_k = st.number_input("Top-k", min_value=0, max_value=1000, value=50, step=1)
with col5:
    num_return_sequences = st.number_input("Return sequences", min_value=1, max_value=5, value=1, step=1)
generate_button = st.button("Generate")

def generate_from_model(prompt_text, model, tokenizer, device, max_new_tokens, temperature, top_k, top_p, num_return_sequences):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=int(num_return_sequences),
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in out]
    return decoded

if generate_button:
    if "model_state" not in st.session_state:
        st.warning("Model not loaded. Set base model and (optionally) adapter path in the sidebar and click 'Load model'.")
    else:
        st.info("Generating...")
        ms = st.session_state.model_state
        try:
            decoded_list = generate_from_model(
                prompt,
                ms["model"],
                ms["tokenizer"],
                ms["device"],
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                num_return_sequences
            )
            for i, d in enumerate(decoded_list, start=1):
                st.markdown(f"**Output {i}:**")
                st.code(d)
                st.download_button(f"Download Output {i}", d.encode("utf-8"), file_name=f"output_{i}.txt")
        except Exception as e:
            st.error(f"Generation failed: {e}")

st.markdown("---")

