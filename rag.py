import re
import uuid
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import lmdeploy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = "BAAI/bge-small-en-v1.5" #"EleutherAI/gpt-j-6B" #"deepseek-ai/DeepSeek-V3"
tokenizer = AutoTokenizer.from_pretrained(model)
rag = AutoModel.from_pretrained(model).to(device)


def tokenize(raw_text, chunk_size):
    """
    Tokenizes document text into small chunks using an LLM tokenizer.
    """

    chunks = {}
    doc = []
    words = raw_text.split()
    current_chunk = []
    chunk_id = 0

    for word in words:
        current_chunk.append(word)

        if len(current_chunk) >= chunk_size:
            chunks[str(chunk_id)] = {"text": " ".join(current_chunk)}
            chunk_id += 1
            current_chunk = []

    if current_chunk:
        chunks[str(chunk_id)] = {"text": " ".join(current_chunk)}

    doc.append(chunks)
    return doc

def get_doc_embeddings(doc):
    """
    Generate embeddings for document chunks using an LLM.
    """
    doc_embeddings = []
    embeddings = {}

    for chunk in doc:
        for id, chunk_text in chunk.items():
            text = chunk_text.get("text")
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                embeds = rag(**inputs).last_hidden_state.mean(dim=1).squeeze()
            normalized_embeds = embeds / torch.norm(embeds, dim=-1, keepdim=True)
            embeddings[id] = normalized_embeds.detach().cpu().numpy()
    
    doc_embeddings.append(embeddings)
    return doc_embeddings

def get_query_embeddings(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    embeddings = rag(**inputs).last_hidden_state.mean(dim=1).squeeze()
    normalized_embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
    
    return normalized_embeddings.detach().cpu().numpy()

def retrieve(query, doc, doc_embeddings):
    """
    Retrieve top-1 most semantically similar text from the document based on query using hybrid search.
    """
    query = query.lower().strip()
    query_embeddings = get_query_embeddings(query)

    # Keyword search
    keyword_scores = {}
    query_words = set(query.split())

    for chunk in doc[0].values():
        chunk_text = chunk['text'].lower()
        chunk_words = set(chunk_text.split())

        common_words = query_words.intersection(chunk_words)
        if common_words:
            keyword_score = len(common_words) / len(chunk_words)
            keyword_scores[chunk['text']] = keyword_score

    # Semantic search
    semantic_scores = {}
    for embed in doc_embeddings:
        for id, chunk_embeddings in embed.items():
            if np.linalg.norm(query_embeddings) == 0 or np.linalg.norm(chunk_embeddings) == 0:
                semantic_score = 0
            else:
                # Cosine similarity for semantic score
                semantic_score = np.dot(chunk_embeddings, query_embeddings) / (
                    np.linalg.norm(chunk_embeddings) * np.linalg.norm(query_embeddings))
            semantic_scores[id] = semantic_score

    # Hybrid scoring
    hybrid_scores = {}
    for chunk_id, chunk_data in doc[0].items():
        chunk_text = chunk_data['text']
        keyword_score = keyword_scores.get(chunk_text, 0)
        semantic_score = semantic_scores.get(chunk_id, 0)

        hybrid_score = 0.5 * keyword_score + 0.5 * semantic_score
        hybrid_scores[chunk_id] = hybrid_score

    sorted_scores = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)[:1]
    top_chunk_id = sorted_scores[0][0]
    response = doc[0][top_chunk_id]['text']

    return response

def get_response(query, raw_text, tokens, embeddings):
    response = retrieve(query, tokens, embeddings)
    return response


if __name__ == "__main__":
    raw_text = "\n\nChildren throw tantrums when they are overwhelmed by strong emotions, overstimulated, or anxious. Young children may have tantrums because of strong emotions, while older children may have tantrums because they don’t know how to express or manage their feelings. Overstimulation can also cause tantrums, as children don’t always recognize when bright lights or loud noises are bothering them. Anxiety can also cause tantrums, such as when a child feels anxious due to an unexpected event, unrealistic demand, or lack of routine.\n \n\nTantrums may happen when kids are tired, hungry, or uncomfortable. They can have a meltdown because they can’t have something they want (like a toy or candy) or can’t get someone to do what they want (like getting a parent to pay attention to them immediately or getting a sibling to give up the tablet). Learning to deal with frustration is a skill that children gain over"
    
    doc = tokenize(raw_text, 50)
    print("doc:", doc[0])
    doc_embeddings = get_doc_embeddings(doc)

    query = "why do toddlers throw tantrums?"
    query_embeddings = get_query_embeddings(query)
    
    response = retrieve(query, doc, doc_embeddings)
    print(response)