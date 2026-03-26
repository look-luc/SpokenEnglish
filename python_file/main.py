import torch
import torch.nn as nn
from overlap_task import model

def main():
    # checking if there is some kind of GPU available before going to the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    vocab_size = 300
    model_discorese = model(vocab_size).to(device)
    data = []

    lr = 0.01

    optimizer = torch.optim.Adam(model_discorese.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for epoch in range(10):
        for input_ids, segment_ids, target in data:
            outputs = model_discorese.forward(input_ids,segment_ids)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return 0

if __name__ == "__main__":
    main()
