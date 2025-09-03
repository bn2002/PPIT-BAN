import os, zipfile
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
INPUT_TSV  = "./data/yeast/dictionary/protein.dictionary.tsv"
OUTPUT_NPZ = "./data/yeast/dictionary/protein_embeddings_esm2.npz"
TMP_NPZ    = OUTPUT_NPZ + ".tmp" 
BATCH_SIZE = 16  # Giảm batch size để tránh OOM trên GPU

# === helper functions ===
_first_write_done = False
def write_batch_npz(npz_tmp_path: str, items: dict[str, np.ndarray]) -> None:
    global _first_write_done
    mode = 'a'
    if not _first_write_done:
        if os.path.exists(npz_tmp_path):
            os.remove(npz_tmp_path)
        mode = 'w'
        _first_write_done = True
    with zipfile.ZipFile(npz_tmp_path, mode=mode, compression=zipfile.ZIP_DEFLATED) as zf:
        for key, arr in items.items():
            with zf.open(f"{key}.npy", 'w') as f:
                np.save(f, arr)

def process_batch_parallel(tokenizer, model, device, batch_data):
    """
    Xử lý một batch các protein sequences song song trên GPU
    batch_data: list of (name, seq) tuples
    """
    names, sequences = zip(*batch_data)
    
    # Tokenize tất cả sequences cùng lúc với padding
    encoded = tokenizer(
        list(sequences), 
        return_tensors="pt", 
        padding=True, 
        truncation=False  # Không truncate, giữ nguyên như code cũ
        # max_length=1024  # Bỏ giới hạn này để giống code cũ
    )
    
    # Chuyển lên GPU
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Forward pass cho toàn bộ batch
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
    
    # Xử lý từng sequence trong batch để tạo embeddings
    batch_embeddings = {}
    for i, name in enumerate(names):
        # Lấy chiều dài thực của sequence (không bao gồm padding)
        mask = attention_mask[i].bool()
        seq_len = mask.sum().item()
        
        # Logic tương đương với code cũ: bỏ [CLS] (đầu) và [SEP]/[PAD] (cuối)
        # Code cũ: [:, 1:-1, :] nghĩa là bỏ token đầu và cuối
        protein_hidden = hidden_states[i, 1:seq_len-1, :]  # [actual_protein_len, hidden_dim]
        
        # Sum pooling giống như code cũ: sum(dim=0)
        protein_embedding = protein_hidden.sum(dim=0).detach().cpu().numpy()  # [hidden_dim]
        batch_embeddings[name] = protein_embedding
    
    return batch_embeddings

# === load model / data ===
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

df = pd.read_csv(INPUT_TSV, sep="\t", header=None, names=["name", "seq"])
print(f"Loaded {len(df)} protein sequences")

# Lọc ra protein cuối cùng cho mỗi tên (để tránh trùng lặp)
last_idx = df.groupby("name").tail(1).reset_index().set_index("name")["index"].to_dict()
unique_proteins = [(row["name"], row["seq"]) for i, row in df.iterrows() if i == last_idx[row["name"]]]
print(f"Found {len(unique_proteins)} unique proteins")

# Xử lý theo batch song song
all_embeddings = {}
batch_data = []

for i, (name, seq) in enumerate(unique_proteins):
    batch_data.append((name, seq))
    
    # Khi đủ batch size hoặc đến cuối list
    if len(batch_data) >= BATCH_SIZE or i == len(unique_proteins) - 1:
        print(f"Processing batch {i//BATCH_SIZE + 1}, proteins {i-len(batch_data)+1} to {i+1}")
        
        # Xử lý batch song song
        batch_embeddings = process_batch_parallel(tokenizer, model, device, batch_data)
        all_embeddings.update(batch_embeddings)
        
        # Ghi batch vào file
        write_batch_npz(TMP_NPZ, batch_embeddings)
        batch_data.clear()
        
        # Dọn dẹp GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        print(f"Saved batch {i//BATCH_SIZE + 1}, total processed: {len(all_embeddings)}")

os.replace(TMP_NPZ, OUTPUT_NPZ)
print(f"Done. NPZ written to: {OUTPUT_NPZ}")
print(f"Total embeddings created: {len(all_embeddings)}")
