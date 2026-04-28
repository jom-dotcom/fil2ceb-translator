import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu

MODEL_OPTIONS = {
    "Base NLLB (No Fine-Tuning)": "facebook/nllb-200-distilled-600M",
    "Model A (Fine-Tuned)": "Jom-0209/nllb-fil2ceb-modelA",
    "Model B (Back-Translation Augmented)": "Jom-0209/nllb-fil2ceb-modelB",
}

loaded_models = {}

def load_model(model_name):
    if model_name not in loaded_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        loaded_models[model_name] = (tokenizer, model, device)
    return loaded_models[model_name]

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

def translate(text, model_choice, show_candidates):
    if not text.strip():
        return "Please enter some text.", ""

    model_name = MODEL_OPTIONS[model_choice]
    tokenizer, model, device = load_model(model_name)

    candidates = get_candidates(text, model, tokenizer, device)
    best = mbr_select(candidates)

    candidates_text = ""
    if show_candidates:
        candidates_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

    return best, candidates_text

with gr.Blocks(title="Filipino to Cebuano Translator") as demo:
    gr.Markdown("# 🇵🇭 Filipino to Cebuano Translator")
    gr.Markdown("Powered by Fine-Tuned NLLB Transformer with MBR Decoding")

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=list(MODEL_OPTIONS.keys()),
            value="Model B (Back-Translation Augmented)",
            label="Select Model"
        )

    text_input = gr.Textbox(
        label="Enter Filipino text:",
        placeholder="Type here...",
        lines=4
    )

    show_candidates = gr.Checkbox(label="Show all MBR candidates")

    translate_btn = gr.Button("Translate", variant="primary")

    translation_output = gr.Textbox(label="Cebuano Translation:")
    candidates_output = gr.Textbox(label="All Candidates:", visible=True)

    translate_btn.click(
        fn=translate,
        inputs=[text_input, model_choice, show_candidates],
        outputs=[translation_output, candidates_output]
    )

demo.launch()