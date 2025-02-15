import os
import json
import spacy

data_dir = "./data/opensubtitles_en_ko"

def download_spacy_model(model_name):
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model: {model_name}")
        os.system(f"python -m spacy download {model_name}")

download_spacy_model("en_core_web_sm")
download_spacy_model("ko_core_news_sm")

spacy_en = spacy.load("en_core_web_sm")
spacy_ko = spacy.load("ko_core_news_sm")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_ko(text):
    return [tok.text for tok in spacy_ko.tokenizer(text)]

def load_dataset():
    src_file = os.path.join(data_dir, "OpenSubtitles.en-ko.en")
    trg_file = os.path.join(data_dir, "OpenSubtitles.en-ko.ko")

    src_texts, trg_texts = [], []
    with open(src_file, "r", encoding="utf-8") as f_en, open(trg_file, "r", encoding="utf-8") as f_ko:
        for en_line, ko_line in zip(f_en, f_ko):
            src_texts.append(en_line.strip())
            trg_texts.append(ko_line.strip())

    return src_texts, trg_texts
