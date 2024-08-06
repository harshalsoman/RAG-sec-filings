import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_llama_model():
    model_name = 'unsloth/llama-3-8b-bnb-4bit'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def implement_rag(all_chunks):
    # Initialize embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create vector database
    embeddings = embed_model.encode(all_chunks)
    dimension = embeddings.shape[1]

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return embed_model, index, all_chunks


def retrieve(query, embed_model, index, chunks, k=5):
    query_vector = embed_model.encode([query])
    faiss.normalize_L2(query_vector)
    _, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]


def generate_response(query, context, tokenizer, model):
    prompt = f"Query: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Generate up to 200 new tokens
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,  # Ensure the model recognizes the end-of-sequence token
            early_stopping=True  # Stop early when an end-of-sequence token is generated
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()
