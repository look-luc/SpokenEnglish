# Updated tokenizer.py
from nltk.tokenize import TreebankWordTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import BertProcessing

class tokenizer:
    def __init__(self, vocab_size=10000):
        self.nltk_tokenizer = TreebankWordTokenizer()
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    def train_on_corpus(self, texts):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        )
        # NLTK pre-tokenization can be used here to clean the corpus
        corpus = [" ".join(self.nltk_tokenizer.tokenize(t)) for t in texts]
        self.tokenizer.train_from_iterator(corpus, trainer=trainer)

        # Configure the tokenizer to add [CLS] and [SEP] automatically like BERT
        self.tokenizer.post_processor = BertProcessing(
            ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
        )

    def encode(self, text1, text2, max_length=512):
        """Encodes a pair of sentences into input_ids and segment_ids."""
        encoding = self.tokenizer.encode(text1, text2)

        # Truncate and pad manually to match max_length
        ids = encoding.ids[:max_length]
        type_ids = encoding.type_ids[:max_length]

        padding_len = max_length - len(ids)
        ids += [self.tokenizer.token_to_id("[PAD]")] * padding_len
        type_ids += [0] * padding_len  # Simplified segment IDs

        return ids, type_ids