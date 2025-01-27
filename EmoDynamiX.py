from modules.roberta import RobertaHeterogeneousGraph
import torch
import numpy as np


def load_full_model(dataset, checkpoint_path):
    class Args:
        exclude_others = 0
        erc_temperature = 0.5
        erc_mixed = 1
        hg_dim = 512

        def __init__(self, dataset):
            self.dataset = dataset

    # load model
    args = Args(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaHeterogeneousGraph(args, lightmode=False)
    model.load(checkpoint_path)
    model.to(device)
    model.eval()
    return model


class EmoDynamiX:
    def __init__(self, dataset, checkpoint_path):
        self.model = load_full_model(dataset, checkpoint_path)

    def make_input(self, d):
        text = " </s> ".join([t['text'] for t in d])
        strategies = [self.model.strategy2id[t['strategy']] if t['speaker'] == 'sys' else -1 for t in d]
        speakers = ["seeker" if t['speaker'] == 'usr' else "supporter" for t in d]
        speakers = " ".join(speakers)
        return {
            "dialogue_history": [text, ],
            "strategy_history": [str(strategies), ],
            "speaker_turn": [speakers, ],
        }

    def predict(self, d):
        model_input = self.make_input(d)
        output = self.model(model_input)
        next_strategy = self.model.id2strategy[np.argmax(output["logits"].cpu().detach().numpy())]
        output["next_strategy"] = next_strategy
        return output
