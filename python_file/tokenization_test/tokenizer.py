import spacy
from spacy.symbols import ORTH

OVERLAP_MAP = {
    r"\[\d*": "<SOV>",  # Matches [, [2, [3, etc.
    r"\]\d*": "<EOV>",  # Matches ], ]2, ]3, etc.
    r"\.\.": "<PAUSE>", # Standard SBC pause marker
    r"--": "<TRUNC>"    # Truncated speech marker
}

class tokenizer:
    def __init__(self,vocab_size=10000):
        self.vocab_size = vocab_size
        self.nlp = spacy.blank("en")

        # <SOV> is start of overlap, <EOV> is end of overlap
        self.special_tokens = ["<SOV>","<EOV>","<PAUSE>","<TRUNC>"]
        for token in self.special_tokens:
            self.nlp.tokenizer.add_special_token(token, [{ORTH: token}])

    def preprocess(self, text):
        for item, replacement in OVERLAP_MAP.items():
            text = text.replace(item, replacement)
        return " ".join(text.split())

    def __call__(self, text):
        clean = self.preprocess(text)
        doc = self.nlp(clean)
        return [token.text for token in doc]