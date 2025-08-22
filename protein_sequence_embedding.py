import os, zipfile
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
INPUT_TSV  = "./data/yeast/dictionary/protein.dictionary.tsv"
OUTPUT_NPZ = "./data/yeast/dictionary/protein_embeddings_esm2.npz"
TMP_NPZ    = OUTPUT_NPZ + ".tmp" 
BATCH_SIZE = 50

# === helper: ghi 1 batch vào .npz (ZIP) ===
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

# === load model / data ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_csv(INPUT_TSV, sep="\t", header=None, names=["name", "seq"])

last_idx = df.groupby("name").tail(1).reset_index().set_index("name")["index"].to_dict()

batch = {}
with torch.inference_mode():
    for i, row in df.iterrows():
        name = row["name"]
        if i != last_idx[name]:
            # có trùng tên
            continue

        seq  = row["seq"]
        toks = tokenizer(seq, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}

        out = model(**toks)                      # [1, L, H]
        emb = out.last_hidden_state[:, 1:-1, :]  # [1, L-2, H]
        emb = emb.sum(dim=0).detach().cpu().numpy()

        batch[name] = emb

        if len(batch) >= BATCH_SIZE:
            write_batch_npz(TMP_NPZ, batch)
            batch.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Saved a batch at index {i}")

    if batch:
        write_batch_npz(TMP_NPZ, batch)
        print("Saved final batch")

os.replace(TMP_NPZ, OUTPUT_NPZ)
print(f"Done. NPZ written to: {OUTPUT_NPZ}")
