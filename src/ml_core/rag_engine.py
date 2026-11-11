import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

os.environ["HF_HOME"] = "C:/Projects/ai/src/ml_core/model_cache"
MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DOCS_DIR = Path("C:/Projects/ai/knowledge_docs").resolve()
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Seed a default corporate compliance policy file if none exists
sample_policy = DOCS_DIR / "company_policies.txt"
if not sample_policy.exists():
    sample_policy.write_text(
        "Policy 101 (Cloud Expenses): Any monthly cloud server cost increase above 15% requires a Forensic Audit and supervisor escalation call.\n"
        "Policy 102 (Vendor Invoices): All vendor invoices over $5,000 must be cross-checked against approved Purchase Orders before payment.\n"
        "Policy 103 (Data Security): Production customer records must never be uploaded to external cloud APIs; use local on-premises models only.\n",
        encoding="utf-8"
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def embed_text(text: str) -> torch.Tensor:
    """Computes a normalized L2 PyTorch embedding tensor locally."""
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls_embedding, p=2, dim=1)

def retrieve_relevant_policy(query: str, top_k: int = 2) -> str:
    """Searches local documents using PyTorch tensor matrix multiplication (Cosine Similarity)."""
    chunks: list[str] = []
    for file_path in DOCS_DIR.glob("*.txt"):
        lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        chunks.extend(lines)

    if not chunks:
        return "No local policy documents found."

    query_vec = embed_text(query)
    doc_vecs = torch.cat([embed_text(chunk) for chunk in chunks], dim=0)
    scores = torch.matmul(doc_vecs, query_vec.T).squeeze(-1)

    k = min(top_k, len(chunks))
    top_indices = torch.topk(scores, k=k).indices.tolist()
    matched = [chunks[idx] for idx in top_indices]
    return " | ".join(matched)
