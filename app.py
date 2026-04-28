import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu

st.set_page_config(page_title="Filipino to Cebuano Translator", page_icon="🇵🇭")
st.title("🇵🇭 Filipino to Cebuano Translator")
st.write("Powered by Fine-Tuned NLLB Transformer with MBR Decoding")

MODEL_OPTIONS = {
    "Base NLLB (No Fine-Tuning)": "facebook/nllb-200-distilled-600M",
    "Model A (Fine-Tuned)": "Jom-0209/nllb-fil2ceb-modelA",
    "Model B (Back-Translation Augmented)": "Jom-0209/nllb-fil2ceb-modelB",
}

selected_label = st.selectbox("Select Model:", list(MODEL_OPTIONS.keys()))

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model, device

with st.spinner("Loading model... this may take a moment."):
    tokenizer, model, device = load_model(MODEL_OPTIONS[selected_label])

def get_candidates(text, model, tokenizer, device, num_candidates=5):
    model.eval()
    inputs = tokenizer(
        text, return_tensors="pt",
        padding=True, truncation=True, max_length=256
    ).to(device)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("ceb_Latn"),
            max_length=256,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            num_return_sequences=num_candidates
        )

    candidates = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return candidates

def mbr_select(candidates):
    best_candidate = None
    best_score = -1

    for i, candidate in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i]
        score = sacrebleu.corpus_chrf([candidate], [[o] for o in others]).score
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate

text = st.text_area("Enter Filipino text:", height=150)
show_candidates = st.checkbox("Show all MBR candidates")

if st.button("Translate"):
    if text.strip():
        with st.spinner("Generating candidates and applying MBR decoding..."):
            candidates = get_candidates(text, model, tokenizer, device)
            best = mbr_select(candidates)

        st.subheader("Cebuano Translation:")
        st.success(best)

        if show_candidates:
            st.subheader("All Candidates:")
            for i, c in enumerate(candidates):
                st.write(f"{i+1}. {c}")
    else:
        st.warning("Please enter some text.")