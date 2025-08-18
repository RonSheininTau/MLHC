
import torch
from transformers import AutoTokenizer, AutoModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
mdl = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)


def chunk_text(text, max_chars=800, overlap=120):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        chunks.append(chunk)
        i += max_chars - overlap
    return chunks


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts

def embed_clinical(texts):
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
    emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
    # L2-normalize for cosine similarity
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

def run_embeeding(notes):
    return notes["text"].apply(lambda x: embed_clinical(chunk_text(x)))