from transformers import AutoTokenizer

# These remain atomic units in the vocabulary
CUSTOM_MARKERS = ["[", "]", "(H)", "[(H)=]", "2", "..."]

class tokenizer():
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(CUSTOM_MARKERS)

    def tokenize_function(self, examples, label2id):
        tokenized = self.tokenizer(
            examples["ut1_text"],
            examples["ut2_text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

        tokenized["labels"] = [label2id[overlap] for overlap in examples["overlap_type"]]
        return tokenized