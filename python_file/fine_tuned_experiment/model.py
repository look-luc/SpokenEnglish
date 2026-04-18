import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoConfig

class overlap_model():
    def __init__(self, model_name="YituTech/conv-bert-base"):
        self.labels = ['recognitional', 'other', 'transitional', 'progressional', 'restatement']
        self.id2label = {idx: label for idx, label in enumerate(self.labels)}
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}

        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )

        # 1. Download the raw weights dictionary directly
        print("Downloading and intercepting raw checkpoint weights...")
        weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")

        # 2. Fix the LayerNorm keys IN THE DICTIONARY before loading
        keys = list(state_dict.keys())
        updated_count = 0
        for key in keys:
            if "LayerNorm.gamma" in key:
                new_key = key.replace("LayerNorm.gamma", "LayerNorm.weight")
                state_dict[new_key] = state_dict.pop(key)
                updated_count += 1
            elif "LayerNorm.beta" in key:
                new_key = key.replace("LayerNorm.beta", "LayerNorm.bias")
                state_dict[new_key] = state_dict.pop(key)
                updated_count += 1

        print(f"Successfully remapped {updated_count} LayerNorm keys from the raw checkpoint.")

        # 3. Load the model using the CORRECTED state_dict
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            state_dict=state_dict,
            ignore_mismatched_sizes=True # Ensures the new classification head initializes cleanly
        )

    def resize_embeddings(self, tokenizer_len):
        self.model.resize_token_embeddings(tokenizer_len)