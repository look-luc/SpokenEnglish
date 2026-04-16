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

    data = pd.read_json("../data/FINAL_DATA_TO_RUN/data_with_edges.json")

    label_map = {'recognitional': 0, 'other': 1, 'transitional': 2, 'progressional': 3}
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
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(
        input_ids_tensor[train_idx],
        segment_ids_tensor[train_idx],
        labels_tensor[train_idx]
    )
    val_dataset = TensorDataset(
        input_ids_tensor[test_idx],
        segment_ids_tensor[test_idx],
        labels_tensor[test_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    actual_vocab_size = custom_tokenizer.tokenizer.get_vocab_size()
    model_discorese = model(vocab_size=actual_vocab_size).to(device)

    class_counts = np.bincount(labels_raw, minlength=4)

    weights = len(labels_raw) / (len(class_counts) * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    lr = 1e-4
    optimizer = torch.optim.Adam(model_discorese.parameters(), lr=lr)

    for epoch in range(10):
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

        print(f"Epoch: {epoch + 1} | Train Loss: {total_train_loss / len(train_loader):.4f}")

        # Validation
        model_discorese.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for b_input_ids, b_segment_ids, b_target in val_loader:
                b_input_ids, b_segment_ids, b_target = b_input_ids.to(device), b_segment_ids.to(device), b_target.to(device)
                outputs = model_discorese(b_input_ids, b_segment_ids)
                _, preds = torch.max(outputs, 1)
                all_true.extend(b_target.cpu().numpy())
                all_pred.extend(preds.cpu().numpy())

        if len(all_true) > 0:
            val_acc = accuracy_score(all_true, all_pred)
            val_f1 = f1_score(all_true, all_pred, average="macro")
            print(f"VALIDATION: acc={val_acc:.3f}, f1={val_f1:.3f}")

    torch.save(model_discorese.state_dict(), "multi_ling_emotion.pth")
    return 0

if __name__ == "__main__":
    main()