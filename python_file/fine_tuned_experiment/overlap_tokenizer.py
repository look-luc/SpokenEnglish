from transformers import AutoTokenizer

import re

OVERLAP_MAP = {
    r'\(\(.*?\)\)': "<NOISE>",
    r'\(H\)=?': "<BREATH>",
    r'\(.*?\)': "<SOUND>",
    r"\[\d+?": "<SOV>",
    r"\d+?\]": "<EOV>",
    r"\[": "<SOV>",
    r"\]": "<EOV>",
    r"\.\.": "<PAUSE>",
    r"--": "<TRUNC>",
}

class tokenizer():
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.custom_special_tokens = list(set(OVERLAP_MAP.values()))
        self.tokenizer.add_tokens(self.custom_special_tokens)

    def preprocess(self, text):
        if not text:
            raise ValueError("Need to provide text")
        for pattern, replacement in OVERLAP_MAP.items():
            text = re.sub(pattern, replacement, text)
        return text

    def tokenize_function(self, examples, label2id):
        utterance1_clean = [self.preprocess(t) for t in examples["ut1_text"]]
        utterance2_clean = [self.preprocess(t) for t in examples["ut2_text"]]

        tokenized = self.tokenizer(
            utterance1_clean,
            utterance2_clean,
            truncation=True,
            max_length=512,
            padding=True,
        )

        tokenized["labels"] = [
            label2id.get(overlap if overlap != "trasitional" else "transitional")
            for overlap in examples["overlap_type"]
        ]
        return tokenized