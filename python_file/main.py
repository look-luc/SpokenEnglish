import torch
import torch.nn as nn
from overlap_task import model
from torch.utils.data import TensorDataset, DataLoader
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # checking if there is some kind of GPU available before going to the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    data = pd.read_json("../data/FINAL_DATA_TO_RUN/data_with_edges.json")
    tags = data["overlap_type"]
    x = data.drop(columns='overlap_type', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x.values,
        tags.values,
        test_size=0.2,
        random_state=42,
        stratify=tags.values  # Keeps class distribution consistent
    )

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    vocab_size = 300
    model_discorese = model(vocab_size).to(device)

    lr = 0.01
    optimizer = torch.optim.Adam(model_discorese.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        print(f"epoch: {epoch+1}")
        for input_ids, segment_ids, target in data:
            outputs = model_discorese.forward(input_ids,segment_ids)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return 0

if __name__ == "__main__":
    main()
    sys.exit(0)
