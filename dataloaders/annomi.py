import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import pickle


class AnnoMIPreProcessed(Dataset):
    def __init__(self, split, args):
        super(AnnoMIPreProcessed, self).__init__()
        self.strategy2id = json.load(open('data/annomi/strategies.json', 'r'))
        if args.exclude_others:
            self.strategy2id = {k: v for k, v in self.strategy2id.items() if k != "Other"}
        self.id2strategy = {v: k for k, v in self.strategy2id.items()}
        self.id2label = self.id2strategy
        assert split in ['train', 'valid', 'test']
        full_data = pickle.load(open(f"data/annomi_preprocessed/{split}.pkl", "rb"))
        class_counts = [0] * len(self.strategy2id.keys())
        self.data = []
        for d in full_data:
            if args.exclude_others and d["label"] == 7:
                continue
            if d["erc_logits"].shape[0] != len(str(d["speaker_turn"]).split(" ")):
                continue
            self.data.append(d)
            class_counts[d["label"]] += 1
        class_weights = [sum(class_counts) / len(class_counts) / c for c in class_counts]
        class_weights = F.softmax(torch.tensor(class_weights) / 1.75)
        self.class_weights = class_weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return {
            "dialogue_history": [b["dialogue_history"] for b in batch],
            "strategy_history": [b["strategy_history"] for b in batch],
            "speaker_turn": [b["speaker_turn"] for b in batch],
            "gold_standard": [b["gold_standard"] for b in batch],
            "label": torch.tensor([b["label"] for b in batch]),
            "parsed_dialogue": [b["parsed_dialogue"] for b in batch],
            "erc_logits": torch.cat([torch.tensor(d["erc_logits"]) for d in batch], dim=0),
        }
