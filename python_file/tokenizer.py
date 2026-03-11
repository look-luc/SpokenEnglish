from nltk.tokenize import TreebankWordTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class NLTK_Tokenizer:
    def __init__(self):
        # initializing type of tokenizer that treats punctuation as tokens
        self.tokenizer = TreebankWordTokenizer()
    def tokenize(
            self,
            #if the pair of sentences are passed as 1 unit, a list of 2 sentences
            sentence_pair:list[str]|None,

            # sentence 1 and 2 are just in case that it is not a list of the two sentence
            sentence_1:str|None=None,
            sentence_2:str|None=None
    )->list[str]:
        '''
        :param sentence_pair:
        :param sentence_1:
        :param sentence_2:
        :return:
        '''
        '''
        Check if the sentences are passed as two separate sentences
        '''
        if sentence_1 is not None and sentence_2 is not None and sentence_pair is None:

            # tokenize the two sentences separately
            tokens_sent1 = self.tokenizer.tokenize(sentence_1)
            tokens_sent2 = self.tokenizer.tokenize(sentence_2)

            # combine the list into one master list
            tokens = tokens_sent1 + tokens_sent2

            # makes the list into a corpus
            corpus = [" ".join(tokens)]

            # using Byte Pair Encoding (BPE) tokenizer
            tokenizer = Tokenizer(
                BPE(
                    unk_token="[UNK]" # unknown token
                )
            )

            # set up training BPE encoding
            trainer = BpeTrainer(
                vocab_size=10000,
                special_tokens=[
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                    "[SEP]"
                ]
            )

            return tokenizer.train(corpus, trainer=trainer) # trained the tokenizer and returned

        # seeing if the list of sentence is passed
        elif sentence_pair is not None and sentence_1 is None and sentence_2 is None:
            tokens = []
            for sentence in sentence_pair: # goes through each sentence
                sub_tokens = self.tokenizer.tokenize(sentence) # tokenize the sentence
                tokens.extend(sub_tokens) # adds all of the tokens to a main token list

            # makes the list into a corpus
            corpus = [" ".join(tokens)]

            # using Byte Pair Encoding (BPE) tokenizer
            tokenizer = Tokenizer(
                BPE(
                    unk_token="[UNK]"
                )
            )
            # set up training BPE encoding
            trainer = BpeTrainer(
                vocab_size=10000,
                special_tokens=[
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                    "[SEP]"
                ]
            )
            return tokenizer.train(corpus, trainer=trainer) # trained the tokenizer and returned
        else:
            # makes sure that only a pair of sentences will be passed
            raise Exception("There has to be some kind of way to tokenize only 2 sentences")