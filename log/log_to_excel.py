import re
import pandas as pd

# Path to your log file
file_path = './pretrained_model/with_edges.log'  # or 'without_edges.log'
output_file = "pretrained_model/extracted_metrics_with.xlsx"

# 1. Patterns to handle different quote/number styles in the logs
# Training logs use strings: {'loss': '0.6360', ..., 'epoch': '1.09'}
train_pattern = r"\{'loss':\s*'([\d.]+)',.*?'epoch':\s*'([\d.]+)'\}"

# Eval logs use floats: {'eval_loss': 0.6462, 'eval_f1': 0.0962, ..., 'epoch': 1.0}
eval_pattern = r"\{'eval_loss':\s*([\d.]+),\s*'eval_f1':\s*([\d.]+),.*?'epoch':\s*([\d.]+)\}"

results = {}

with open(file_path, 'r') as f:
    content = f.read()

    # Extract Training Data
    # Since training logs happen multiple times per epoch,
    # we update the dictionary so the last 'train' entry before an eval is kept.
    train_matches = re.findall(train_pattern, content)
    for loss, epoch_str in train_matches:
        epoch_num = int(float(epoch_str)) + 1 # Aligning 0.99 or 1.09 to Epoch 1
        results[epoch_num] = {
            "Epoch": epoch_num,
            "Train Loss": float(loss),
            "Val Loss": None,
            "Val F1": None
        }

    # Extract Evaluation Data and merge with existing Epoch entries
    eval_matches = re.findall(eval_pattern, content)
    for val_loss, val_f1, epoch_val in eval_matches:
        epoch_num = int(float(epoch_val))
        if epoch_num not in results:
            results[epoch_num] = {"Epoch": epoch_num, "Train Loss": None}

        results[epoch_num].update({
            "Val Loss": float(val_loss),
            "Val F1": float(val_f1)
        })

# 2. Prepare for Excel
# Sort by epoch and convert to list
sorted_data = [results[k] for k in sorted(results.keys())]

df = pd.DataFrame(sorted_data)
df.to_excel(output_file, index=False)

print(f"Excel file '{output_file}' has been created with {len(df)} epochs.")