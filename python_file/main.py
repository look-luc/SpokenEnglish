import torch

def main():
    # checking if there is some kind of GPU available before going to the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return 0

if __name__ == "__main__":
    main()
