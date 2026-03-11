from nltk.tokenize import TreebankWordTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class NLTK_Tokenizer:
    def __init__(self):
        self.tokenizer = TreebankWordTokenizer()
    def tokenize(self, sentence_pair:list[str]|None, sentence_1:str|None=None, sentence_2:str|None=None)->list[str]:
        if sentence_1 is not None and sentence_2 is not None and sentence_pair is None:
            tokens_sent1 = self.tokenizer.tokenize(sentence_1)
            tokens_sent2 = self.tokenizer.tokenize(sentence_2)
            tokens = tokens_sent1 + tokens_sent2

            corpus = [" ".join(tokens)]
            tokenizer = Tokenizer(
                BPE(
                    unk_token="[UNK]"
                )
            )
            trainer = BpeTrainer(
                vocab_size=10000,
                special_tokens=[
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                    "[SEP]"
                ]
            )

            return tokenizer.train(corpus, trainer=trainer)
        elif sentence_pair is not None and sentence_1 is None and sentence_2 is None:
            tokens = []
            for sentence in sentence_pair:
                sub_tokens = self.tokenizer.tokenize(sentence)
                tokens.extend(sub_tokens)

            corpus = [" ".join(tokens)]
            tokenizer = Tokenizer(
                BPE(
                    unk_token="[UNK]"
                )
            )
            trainer = BpeTrainer(
                vocab_size=10000,
                special_tokens=[
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                    "[SEP]"
                ]
            )
            return tokenizer.train(corpus, trainer=trainer)
        else:
            raise Exception("There has to be some kind of way to tokenize only 2 sentences")