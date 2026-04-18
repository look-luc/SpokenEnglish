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

        print("Downloading and intercepting raw checkpoint weights...")
        weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        raw_state_dict = torch.load(weights_path, map_location="cpu")

        fixed_state_dict = {}
        updated_count = 0

        for key, value in raw_state_dict.items():
            new_key = key
            lowered_key = key.lower()

            if lowered_key.endswith(".gamma"):
                new_key = key[:key.rfind(".")] + ".weight"
                updated_count += 1
            elif lowered_key.endswith(".beta"):
                new_key = key[:key.rfind(".")] + ".bias"
                updated_count += 1

            fixed_state_dict[new_key] = value

        print(f"Successfully remapped {updated_count} LayerNorm keys from the raw checkpoint.")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )

        load_result = self.model.load_state_dict(fixed_state_dict, strict=False)
        print(f"Weights loaded. Missing keys in checkpoint: {len(load_result.missing_keys)}")

    def resize_embeddings(self, tokenizer_len):
        self.model.resize_token_embeddings(tokenizer_len)