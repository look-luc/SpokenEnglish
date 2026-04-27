from nltk.tokenize import TreebankWordTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import BertProcessing
import re

OVERLAP_MAP = {
    r'\(\(.*?\)\)': "<NOISE>",   # Check for double parens first
    r'\(H\)=?': "<BREATH>",
    r'\(.*?\)': "<SOUND>",      # Then single parens
    r"\[\d+": "<SOV>",
    r"\d+\]": "<EOV>",
    r"\.\.": "<PAUSE>",
    r"--": "<TRUNC>",
}

class tokenizer:
    def __init__(self, vocab_size=10000):
        self.nltk_tokenizer = TreebankWordTokenizer()
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    def train_on_corpus(self, texts):
        trainer = BpeTrainer(vocab_size=self.vocab_size,
                             special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + list(OVERLAP_MAP.values()))

        processed_texts = []
        for text in texts:
            temp_text = text
            for pattern, replacement in OVERLAP_MAP.items():
                temp_text = re.sub(pattern, replacement, temp_text)
            processed_texts.append(" ".join(self.nltk_tokenizer.tokenize(temp_text)))

        self.tokenizer.train_from_iterator(processed_texts, trainer=trainer)

        cls_id = self.tokenizer.token_to_id("[CLS]")
        sep_id = self.tokenizer.token_to_id("[SEP]")
        self.tokenizer.post_processor = BertProcessing(
            ("[SEP]", sep_id),
            ("[CLS]", cls_id),
        )

    def encode(self, text1, text2, max_length=512):
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), length=max_length)
        self.tokenizer.enable_truncation(max_length=max_length)

        encoding = self.tokenizer.encode(text1, text2)
        return encoding.ids, encoding.type_ids