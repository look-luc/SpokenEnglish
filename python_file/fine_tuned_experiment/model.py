from transformers import AutoModelForSequenceClassification, AutoConfig

class overlap_model():
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        self.labels = ['recognitional', 'other', 'transitional', 'progressional', 'restatement']
        self.id2label = {idx: label for idx, label in enumerate(self.labels)}
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}

        # Load config first to ensure consistency
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )

        self._fix_layernorm_keys()

    def _fix_layernorm_keys(self):
        """Maps 'gamma' to 'weight' and 'beta' to 'bias' for LayerNorm stability."""
        state_dict = self.model.state_dict()
        loaded_keys = list(state_dict.keys())
        new_state_dict = {}

        for key in loaded_keys:
            new_key = key.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias")
            new_state_dict[new_key] = state_dict[key]

        self.model.load_state_dict(new_state_dict, strict=False)

    def resize_embeddings(self, tokenizer_len):
        self.model.resize_token_embeddings(tokenizer_len)