import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import fitz  # PyMuPDF for PDF reading

# Cache model loading
@st.cache_resource
def load_summarizer(model_name):
    if model_name == "allenai/led-base-16384":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return {"tokenizer": tokenizer, "model": model}
    else:
        return pipeline("summarization", model=model_name)

# LED summarization function
def summarize_with_led(text, tokenizer, model, min_length, max_length):
    inputs = tokenizer(text, return_tensors="pt", max_length=16384, truncation=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1  # global attention on first token

    summary_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
        min_length=min_length,
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Topic filtering function
def extract_relevant_text(text, topic):
    sentences = text.split(".")
    filtered = [s.strip() for s in sentences if topic.lower() in s.lower()]
    return ". ".join(filtered) + "." if filtered else text  # fallback to full text

# App title
st.title("🧠 NeuroBrief")

# Model selection
model_choice = st.selectbox(
    "Choose a model:",
    (
        "facebook/bart-large-cnn (English)",
        "csebuetnlp/mT5_multilingual_XLSum (Multilingual)",
        "allenai/led-base-16384 (Longformer for Long Docs)"
    )
)

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
user_text = ""

# Extract text from PDF
if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()
        user_text = pdf_text
        st.success("Text extracted from PDF:")
        st.text_area("Extracted Text", user_text, height=250)
else:
    user_text = st.text_area("Or enter text manually:", height=250)

# Topic input
topic_query = st.text_input("Enter a topic or keyword for focused summarization (optional):")

# Summary length sliders
min_length = st.slider("Minimum summary length", min_value=10, max_value=200, value=30)
max_length = st.slider("Maximum summary length", min_value=50, max_value=300, value=130)

# Summarize
if st.button("Generate Summary"):
    if not user_text.strip():
        st.warning("Please enter some text or upload a PDF.")
    elif min_length > max_length:
        st.error("Minimum length should be less than or equal to maximum length.")
    else:
        with st.spinner("Summarizing..."):
            input_text = user_text
            if topic_query.strip():
                input_text = extract_relevant_text(user_text, topic_query.strip())

            if "facebook" in model_choice:
                model = load_summarizer("facebook/bart-large-cnn")
                summary = model(input_text, min_length=min_length, max_length=max_length, do_sample=False)[0]['summary_text']

            elif "mT5" in model_choice:
                model = load_summarizer("csebuetnlp/mT5_multilingual_XLSum")
                summary = model(input_text, min_length=min_length, max_length=max_length, do_sample=False)[0]['summary_text']

            elif "Longformer" in model_choice:
                resources = load_summarizer("allenai/led-base-16384")
                summary = summarize_with_led(input_text, resources["tokenizer"], resources["model"], min_length, max_length)

            else:
                summary = "Model not recognized."

            st.success("Summary:")
            st.write(summary)
