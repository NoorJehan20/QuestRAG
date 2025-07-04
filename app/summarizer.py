from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

def summarize_pdf(pages):
    full_text = " ".join(page.page_content for page in pages)

    model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "saved_model", "flan-t5-base"))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    summary = summarizer(full_text[:2000])[0]['generated_text'] 

    return summary