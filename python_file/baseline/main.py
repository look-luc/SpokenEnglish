import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from overlap_task import model
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import tokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data = pd.read_json("../../data/FINAL_DATA_TO_RUN/data_with_edges.json")

    label_map = {
        'recognitional': 0,
        'other': 1,
        'transitional': 2,
        'progressional': 3,
        'restatement': 4
    }
    labels_raw = data["overlap_type"].map(label_map).fillna(1).astype(int).values

    custom_tokenizer = tokenizer(vocab_size=5000)
    all_text = data['ut1_text'].tolist() + data['ut2_text'].tolist()
    custom_tokenizer.train_on_corpus(all_text)

    all_input_ids = []
    all_segment_ids = []

    for _, row in data.iterrows():
        ids, segs = custom_tokenizer.encode(row['ut1_text'], row['ut2_text'])
        all_input_ids.append(ids)
        all_segment_ids.append(segs)

    input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    segment_ids_tensor = torch.tensor(all_segment_ids, dtype=torch.long)
    labels_tensor = torch.tensor(labels_raw, dtype=torch.long)

    indices = range(len(labels_tensor))
    train_idx, dev_idx = train_test_split(
        np.arange(len(input_ids_tensor)),
        test_size=0.1,
        random_state=42,
        stratify=labels_raw  # Ensure classes are balanced
    )

    train_loader = DataLoader(
        TensorDataset(input_ids_tensor[train_idx], segment_ids_tensor[train_idx], labels_tensor[train_idx]),
        batch_size=32, shuffle=True)
    dev_loader = DataLoader(
        TensorDataset(
            input_ids_tensor[dev_idx],
            segment_ids_tensor[dev_idx],
            labels_tensor[dev_idx]
        ),
        batch_size=32,
        shuffle=False
    )

    actual_vocab_size = custom_tokenizer.tokenizer.get_vocab_size()

    pad_id = custom_tokenizer.tokenizer.token_to_id("[PAD]")

    model_discorese = model(vocab_size=actual_vocab_size, pad_token_id=pad_id).to(device)

    class_counts = np.bincount(labels_raw, minlength=5)

    safe_counts = np.where(class_counts > 0, class_counts, 1)
    weights = len(labels_raw) / (len(class_counts) * safe_counts)

    weights[class_counts == 0] = 0.0

    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    lr = 2e-5
    optimizer = torch.optim.AdamW(model_discorese.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_dev_f1 = -math.inf
    for epoch in range(15):
        model_discorese.train()
        total_train_loss = 0
        for b_input_ids, b_segment_ids, b_target in train_loader:
            b_input_ids, b_segment_ids, b_target = b_input_ids.to(device), b_segment_ids.to(device), b_target.to(device)

            optimizer.zero_grad()
            outputs = model_discorese(b_input_ids, b_segment_ids)
            loss = criterion(outputs, b_target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # --- VALIDATION PHASE ---
        model_discorese.eval()
        best_dev_f1 = 0
        val_loss = 0
        all_true, all_pred = [], []

        with torch.no_grad():
            for b_input_ids, b_segment_ids, b_target in dev_loader:
                b_input_ids, b_segment_ids, b_target = b_input_ids.to(device), b_segment_ids.to(device), b_target.to(
                    device)

                outputs = model_discorese(b_input_ids, b_segment_ids)
                loss = criterion(outputs, b_target)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_true.extend(b_target.cpu().numpy())
                all_pred.extend(preds.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(dev_loader)
        val_f1 = f1_score(all_true, all_pred, average="macro")

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.3f}")

        if val_f1 > best_dev_f1:
            best_dev_f1 = val_f1
            torch.save(model_discorese.state_dict(), "best_model.pt")
        scheduler.step(best_dev_f1)

    return 0

if __name__ == "__main__":
    main()