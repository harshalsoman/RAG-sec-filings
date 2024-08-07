import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import faiss
from data_processing import save_embeddings, load_embeddings
import os

MODEL_DIR = "saved_models/roberta"

def save_model(model, tokenizer, model_dir=MODEL_DIR):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

def load_roberta():
    model_name = 'roberta-base'
    if os.path.exists(MODEL_DIR):
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        save_model(model, tokenizer)
    return model, tokenizer


def generate_response_roberta(query, context, tokenizer, model):
    prompt = f"Query: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Generate up to 200 new tokens
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()