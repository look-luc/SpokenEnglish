import spacy
from spacy.symbols import ORTH
import re

OVERLAP_MAP = {
    r'\(\(.*?\)\)': "{NOISE}",
    r'\(H\)=?': "{BREATH}",
    r'\(.*?\)': "{SOUND}",
    r"\[\d+": "{SOV}",
    r"\d+\]": "{EOV}",
    r"\.\.": "{PAUSE}",
    r"--": "{TRUNC}",
    r'\<': "{START_OTHER}",
    r'\>': "{END_OTHER}",
}


class tokenizer:
    def __init__(self, vocab_size=10000, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.nlp = spacy.blank("en")

        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}

        custom_special_tokens = list(OVERLAP_MAP.values())
        all_special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + custom_special_tokens

        for token in all_special:
            self.nlp.tokenizer.add_special_case(token, [{ORTH: token}])
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def build_vocab(self, all_texts):
        for text in all_texts:
            tokens = self.get_tokens(text)
            for t in tokens:
                if t not in self.vocab and len(self.vocab) < self.vocab_size:
                    self.vocab[t] = len(self.vocab)

    def preprocess(self, text):
        # Apply substitutions in the dictionary order
        for pattern, replacement in OVERLAP_MAP.items():
            text = re.sub(pattern, replacement, text)
        for token in OVERLAP_MAP.values():
            text = text.replace(token, f" {token} ")
        return " ".join(text.split())

    def encode(self, text1, text2):
        tokens1 = self.get_tokens(text1)
        tokens2 = self.get_tokens(text2)

        combined_tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]

        segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)

        input_ids = [self.vocab.get(t, self.vocab["[UNK]"]) for t in combined_tokens]

        input_ids = input_ids[:self.max_length]
        segment_ids = segment_ids[:self.max_length]

        # Padding
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.vocab["[PAD]"]] * padding_length
            segment_ids += [0] * padding_length

        return input_ids, segment_ids

    def get_tokens(self, text):
        if not text: return []
        clean = self.preprocess(text)
        doc = self.nlp(clean)
        return [token.text for token in doc]

    def __call__(self, text_input):
        if isinstance(text_input, list):
            return [self.get_tokens(t) for t in text_input]
        return self.get_tokens(text_input)

    def get_vocab_size(self):
        return len(self.vocab)