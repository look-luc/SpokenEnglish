import spacy
from spacy.symbols import ORTH
import re

OVERLAP_MAP = {
    r"\[\d*": "<SOV>",  # Matches [, [2, [3, etc.
    r"\]\d*": "<EOV>",  # Matches ], ]2, ]3, etc.
    r"\.\.": "<PAUSE>", # Standard SBC pause marker
    r"--": "<TRUNC>"    # Truncated speech marker
}

class tokenizer:
    def __init__(self,vocab_size=10000,max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.nlp = spacy.blank("en")

        # <SOV> is start of overlap, <EOV> is end of overlap
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.special_tokens = ["<SOV>","<EOV>","<PAUSE>","<TRUNC>"]
        for token in self.special_tokens:
            self.nlp.tokenizer.add_special_case(token, [{ORTH: token}])
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def build_vocab(self, all_texts):
        """Goes through the text and adds to the vocab dictionary"""
        for text in all_texts:
            tokens = self.__call__(text)
            for t in tokens:
                if t not in self.vocab and len(self.vocab) < self.vocab_size:
                    self.vocab[t] = len(self.vocab)

    def preprocess(self, text):
        # FIX 2: Use re.sub to actually process the regex patterns
        for pattern, replacement in OVERLAP_MAP.items():
            text = re.sub(pattern, replacement, text)
        return " ".join(text.split())

    def encode(self, text1, text2):
        """Converts two utterances into a single sequence of IDs and Segment IDs."""
        tokens1 = self.__call__(text1)
        tokens2 = self.__call__(text2)

        # Build sequence: [CLS] Speaker1 [SEP] Speaker2 [SEP]
        combined_tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]

        # Segment IDs: 0 for Speaker 1, 1 for Speaker 2
        segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)

        # Convert to IDs
        input_ids = [self.vocab.get(t, self.vocab["[UNK]"]) for t in combined_tokens]

        # Truncate or Pad to self.max_length
        input_ids = input_ids[:self.max_length]
        segment_ids = segment_ids[:self.max_length]

        padding_length = self.max_length - len(input_ids)
        input_ids += [self.vocab["[PAD]"]] * padding_length
        segment_ids += [0] * padding_length  # Padding usually belongs to segment 0

        return input_ids, segment_ids

    def get_tokens(self, text):
        clean = self.preprocess(text)
        doc = self.nlp(clean)
        return [token.text for token in doc]

    def __call__(self, text_input):
        # FIX 3: Support both single strings and lists of strings
        if isinstance(text_input, list):
            return [self.get_tokens(t) for t in text_input]
        return self.get_tokens(text_input)

    def get_vocab_size(self):
        return len(self.vocab)