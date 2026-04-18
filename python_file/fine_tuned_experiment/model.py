from transformers import AutoModelForSequenceClassification

class overlap_model():
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        self.labels = ['recognitional', 'other', 'transitional', 'progressional', 'restatement']
        self.id2label = {idx: label for idx, label in enumerate(self.labels)}
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )

    def resize_embeddings(self, tokenizer_len):
        """Adjusts the model's internal embedding matrix to match the new vocab size."""
        self.model.resize_token_embeddings(tokenizer_len)