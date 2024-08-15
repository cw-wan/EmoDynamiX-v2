import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from modules.decoder import RobertaClassificationHead


class SequentialERC(nn.Module):
    def __init__(self, args=None):
        super(SequentialERC, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.encoder = RobertaModel.from_pretrained('roberta-base')
        self.roberta_config = RobertaConfig.from_pretrained('roberta-base')
        self.classifier = RobertaClassificationHead(hidden_size=self.roberta_config.hidden_size, num_labels=7)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, samples):
        texts = samples['texts']
        sep_indices = []
        tokenized = (self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True))
        for ids in tokenized['input_ids']:
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            sep_indices.append([i for i, token in enumerate(tokens) if token == '</s>'][:-1])
        embeds = []
        outputs = self.encoder(input_ids=tokenized["input_ids"].to(self.device),
                               attention_mask=tokenized["attention_mask"].to(self.device))
        for i in range(outputs.last_hidden_state.size(0)):
            embeds.append(outputs.last_hidden_state[i][sep_indices[i], :])
        embeds = torch.cat(embeds, dim=0)
        logits = self.softmax(self.classifier(embeds))
        return {
            "logits": logits,
            'embeddings': embeds,
        }

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
