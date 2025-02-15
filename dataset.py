import torch
from torch.utils.data import Dataset
import json
import os
from data_loader import tokenize_en, tokenize_ko

data_dir = "./data/opensubtitles_en_ko"

def save_data(data, path):
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

def load_vocab():
    src_vocab = json.load(open(os.path.join(data_dir, "src_vocab.json"), "r", encoding="utf-8"))
    trg_vocab = json.load(open(os.path.join(data_dir, "trg_vocab.json"), "r", encoding="utf-8"))
    return src_vocab, trg_vocab

class TranslationDataset(Dataset):
    def __init__(self, src_texts, trg_texts, src_vocab, trg_vocab, max_len=50):
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_seq = self.text_to_tensor(self.src_texts[idx], self.src_vocab)
        trg_seq = self.text_to_tensor(self.trg_texts[idx], self.trg_vocab)
        return src_seq, trg_seq
    
    def text_to_tensor(self, text, vocab):
        tokens = tokenize_en(text) if vocab == src_vocab else tokenize_ko(text)
        token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        token_ids = token_ids[:self.max_len] + [vocab["<PAD>"]] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)
