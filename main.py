# main.py
import torch
torch.classes.__path__ = []  # REQUIRED

from recommendation_engine.api import create_api

app = create_api()
