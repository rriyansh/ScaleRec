import faiss
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.synthetic import generate
from src.data.preprocess import preprocess
from src.retrieval.train import train as train_retrieval
from src.retrieval.build_index import build_index


def test_retrieval_pipeline(cfg_path):
    generate(cfg_path)
    preprocess(cfg_path)
    train_retrieval(cfg_path)
    build_index(cfg_path)

    index = faiss.read_index("models_store/faiss_index.bin")
    cfg = load_config(cfg_path)
    assert index.ntotal == cfg.data.num_items
