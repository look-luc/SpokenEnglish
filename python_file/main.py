import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from overlap_task import model
from torch.utils.data import TensorDataset, DataLoader
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import tokenizer

def main():
    # checking if there is some kind of GPU available before going to the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    actual_vocab_size = custom_tokenizer.tokenizer.get_vocab_size()
    model_discorese = model(vocab_size=actual_vocab_size).to(device)

    optimizer = torch.optim.Adam(model_discorese.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model_discorese.train()
    for epoch in range(10):
        total_loss = 0
        for b_input_ids, b_segment_ids, b_target in train_loader:
            b_input_ids = b_input_ids.to(device)
            b_segment_ids = b_segment_ids.to(device)
            b_target = b_target.to(device)

            optimizer.zero_grad()
            outputs = model_discorese(b_input_ids, b_segment_ids)

            loss = criterion(outputs, b_target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch: {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")

        print("\nStarting validation...")
        model_discorese.eval()

        global_true = []
        global_pred = []
        with torch.no_grad():
            for b_input_ids, b_segment_ids, b_target in train_loader:
                local_true, local_pred = [], []

                b_input_ids = b_input_ids.to(device)
                b_segment_ids = b_segment_ids.to(device)
                b_target = b_target.to(device)

                outputs = model_discorese(b_input_ids, b_segment_ids)
                _, preds = torch.max(outputs, 1)

                local_true.extend(b_target.cpu().numpy())
                local_pred.extend(preds.cpu().numpy())

                if len(local_true) > 0:
                    acc = accuracy_score(local_true, local_pred)
                    f1 = f1_score(local_true, local_pred, average="macro")

                    print(f"Metrics: acc={acc:.3f}, f1={f1:.3f}")

                    global_true.extend(local_true)
                    global_pred.extend(local_pred)
        if len(global_true) > 0:
            global_acc = accuracy_score(global_true, global_pred)
            global_f1 = f1_score(global_true, global_pred, average="macro")

            print(f"\nGLOBAL: acc={global_acc:.3f}, f1={global_f1:.3f}")
    torch.save(model_discorese.state_dict(), "multi_ling_emotion.pth")
    print("\nModel saved as multi_ling_emotion.pth")

    return 0

if __name__ == "__main__":
    main()
    sys.exit(0)
