import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)
mdl = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device).eval()

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                    # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                          # (B, 1)
    return summed / counts

@torch.no_grad()
def embed_long_texts(
    texts,
    batch_size: int = 16,
    max_length: int = 256,
    stride: int = 64,
    use_cls: bool = False,             # True = a bit faster, False = mean-pooled (often better)
    normalize: bool = True
):
    """
    Returns a torch.Tensor of shape (len(texts), hidden_size)
    Uses token-level sliding windows with overflow + stride; averages windows per document.
    """
    # Pre-tokenize with overflowing windows so we avoid manual char-chunking
    all_input_ids, all_attn, doc_ids = [], [], []
    for doc_id, t in enumerate(texts):
        enc = tok(
            t,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding=False
        )
        n_chunks = len(enc["input_ids"])
        all_input_ids.extend(enc["input_ids"])
        all_attn.extend(enc["attention_mask"])
        doc_ids.extend([doc_id] * n_chunks)

    # Nothing to do?
    if len(all_input_ids) == 0:
        hsz = mdl.config.hidden_size
        out = torch.zeros((len(texts), hsz), dtype=torch.float32)
        return out

    # Pad in mini-batches on the fly to the longest within the batch
    def _pad_batch(ids_batch, attn_batch):
        max_len = max(len(x) for x in ids_batch)
        ids_pad = [x + [tok.pad_token_id] * (max_len - len(x)) for x in ids_batch]
        attn_pad = [x + [0] * (max_len - len(x)) for x in attn_batch]
        return (
            torch.tensor(ids_pad, dtype=torch.long),
            torch.tensor(attn_pad, dtype=torch.long),
        )

    embeddings = []
    owners = []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else torch.cuda.amp.autocast(enabled=False)
    )
    with torch.inference_mode(), amp_ctx:
        for i in range(0, len(all_input_ids), batch_size):
            ids_batch = all_input_ids[i:i+batch_size]
            attn_batch = all_attn[i:i+batch_size]
            owner_batch = doc_ids[i:i+batch_size]

            input_ids, attention_mask = _pad_batch(ids_batch, attn_batch)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            out = mdl(input_ids=input_ids, attention_mask=attention_mask)
            if use_cls:
                emb = out.last_hidden_state[:, 0]                 # [CLS]
            else:
                emb = mean_pool(out.last_hidden_state, attention_mask)

            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            embeddings.append(emb.detach().to("cpu"))
            owners.extend(owner_batch)

    # Stack chunk embeddings and average per original document
    chunk_embs = torch.cat(embeddings, dim=0)     # (Nchunks, H)
    owners = torch.tensor(owners, dtype=torch.long)
    H = chunk_embs.size(1)
    doc_sums = torch.zeros((len(texts), H), dtype=chunk_embs.dtype)
    doc_cnts = torch.zeros((len(texts), 1), dtype=chunk_embs.dtype)

    doc_sums.index_add_(0, owners, chunk_embs)
    doc_cnts.index_add_(0, owners, torch.ones((owners.size(0), 1), dtype=chunk_embs.dtype))
    doc_embs = doc_sums / doc_cnts.clamp(min=1e-9)

    return doc_embs  # torch.Tensor on CPU

# ---------- Example Pandas integration ----------
# notes["emb"] = embed_long_texts(notes["text"].tolist(), batch_size=32, max_length=256, stride=64).tolist()