import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def preprocess(dir):
    dialogues = []
    texts = []
    labels = []
    dataset = dir.split('/')[-1]

    text_file = 'dialogues_' + dataset + '.txt'
    emotion_file = 'dialogues_emotion_' + dataset + '.txt'

    with open(os.path.join(dir, text_file), encoding="utf-8") as f:
        dialogues = f.read().splitlines()

    with open(os.path.join(dir, emotion_file), encoding="utf-8") as f:
        emotions = f.read().splitlines()

    for dialogue, emotion in tqdm(zip(dialogues, emotions)):
        turns = dialogue.split("__eou__")[:-1]
        label = emotion.split()
        text = ""
        sample_label = []
        for i in range(len(turns)):
            text += " </s> " + turns[i]
            sample_label.append(label[i])
            # Set 5 as max dialogue turns
            if (i + 1) % 5 == 0 or i == len(turns) - 1:
                texts.append(text)
                labels.append(sample_label)
                text = ""
                sample_label = []

    return texts, labels


class DailyDialogue(Dataset):
    def __init__(self, split, args=None):
        super(DailyDialogue, self).__init__()
        assert split in ["train", "valid", "test"]
        data_path = f"data/dailydialogue/{split}"
        texts, labels = preprocess(data_path)

        self.id2label = {0: 'Neutral', 1: 'Anger', 2: 'Disgust', 3: 'Fear', 4: 'Joy', 5: 'Sadness', 6: 'Surprise'}

        class_count = [0] * 7
        for label in labels:
            for i in label:
                class_count[int(i)] += 1
        self.class_weights = [0] * len(class_count)
        for i in range(len(class_count)):
            self.class_weights[i] = 1 - class_count[i] / sum(class_count)
        self.data = []
        for i in range(len(texts)):
            sample = {
                "text": texts[i],
                "label": [int(lb) for lb in labels[i]],
            }
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return {
            "texts": [d["text"] for d in batch],
            "label": torch.cat([torch.tensor(d["label"]) for d in batch], dim=0),
        }
