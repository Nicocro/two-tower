from collections import Counter
import re 
from typing import List, Dict, Optional
from nltk.stem import PorterStemmer 
import json

class W2VTokenizer():
  def __init__(self, min_freq: int=3, use_stemming: bool = False) -> None:
    self.min_freq = min_freq
    self.use_stemming = use_stemming
    self.vocab = {} # word to indexing mapping
    self.reverse_vocab = {} # index to word inverse mapping
    self.word_counts = Counter()
    self.stemmer = PorterStemmer() if use_stemming else None 

    # Special tokens with their assigned IDs
    self.special_tokens = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<COMMA>': 2,
        '<PERIOD>': 3,
        '<QUOTATION_MARK>': 4,
        '<SEMICOLON>': 5,
        '<EXCLAMATION_MARK>': 6,
        '<QUESTION_MARK>': 7,
        '<LEFT_PAREN>': 8,
        '<RIGHT_PAREN>': 9,
        '<HYPHENS>': 10,
        '<COLON>': 11
    }

  def preprocess_text(self, text: str) -> List[str]:
    """
    basic preprocessing step. Returns a list of tokens 
    """

    text = text.lower() #turn everything to lowercase
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,"!?();:-]', ' ', text)  # Keep all relevant punctuation
    # Then do all the replacements
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace('"',  ' <QUOTATION_MARK> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('(',  ' <LEFT_PAREN> ')
    text = text.replace(')',  ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace(':',  ' <COLON> ')
    
    # text = ' '.join(text.split()) # normalize whitespaces, remove newlines, tabs, extra spaces
    words = text.split()

    if self.use_stemming:
      words = [self.stemmer.stem(word) if not word.startswith('<') else word for word in words]


    return words


  def fit(self, texts: List[str]) -> None:
    """
    Build the vocabulary from a list of training texts
    """
    # count all tokens across all texts
    for text in texts:
      tokens = self.preprocess_text(text)
      self.word_counts.update(tokens)

    # initialize vocabulary with special tokens
    self.vocab = self.special_tokens.copy()
    self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    # add frequent tokens to vocabulary 
    current_idx = len(self.vocab)
    for word, count in self.word_counts.items():
      if count >= self.min_freq:
        self.vocab[word] = current_idx
        self.reverse_vocab[current_idx] = word
        current_idx += 1

  def load_vocab(self, vocab_file: str) -> None:
    """ Load a pre-built vocabulary from a file
    """
    with open(vocab_file, 'r') as f:
      self.vocab = json.load(f)
      self.reverse_vocab = {v: k for k, v in self.vocab.items()}
  
  def tokenize(self, text: str) -> List[int]:
    """ convert a single text string into a List of token IDS
    """
    if not self.vocab:
      raise ValueError("Vocabulary not built yet! Call FIt() first")

    tokens = self.preprocess_text(text)
    return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
    
  def decode(self, token_ids: List[int]) -> List[str]:
    """ convert a List of token IDS into a List of tokens
    """
    return [self.reverse_vocab.get(id, '<UNK>') for id in token_ids]