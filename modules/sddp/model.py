import numpy as np
import torch
import torch.nn as nn
from .struct_attention import Struct_Attention


class Model(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.encoder = kwargs['encoder']
        self.struct_attention = Struct_Attention(config, kwargs['link_only'])

    def encoder_inference(self, mapping):
        encoder_cache = {}
        context_pair_input_ids = []
        context_pair_input_masks = []
        type_ids = []
        keys = []
        for k, v in mapping.items():
            context_pair_input_ids.append(v[0].split())
            context_pair_input_masks.append(v[1])
            type_ids.append(v[2])
            keys.append(k)
        context_pair_input_ids = torch.LongTensor(np.array(context_pair_input_ids).astype(int)).to(self.encoder.device)
        context_pair_input_masks = torch.LongTensor(context_pair_input_masks).to(self.encoder.device)
        type_ids = torch.LongTensor(type_ids).to(self.encoder.device)
        bs = 2000  # depends on gpu mem
        length = context_pair_input_ids.shape[0]
        struct_vecs = []
        for i in range(length // bs + 1):
            if i * bs != length:
                res = self.encoder(context_pair_input_ids[i * bs:(i + 1) * bs],
                                   context_pair_input_masks[i * bs:(i + 1) * bs])[0][:, 0, :]
                struct_vecs.append(res.cpu())
        struct_vecs = torch.cat(struct_vecs, 0)
        for key, sv in zip(keys, struct_vecs):
            encoder_cache[key] = sv

        return encoder_cache

    def inference_forward(self, struct_vec, context_sentence_masks, max_sent_len):
        batch_size = context_sentence_masks.shape[0]
        # distribute result into batch
        struct_vec = struct_vec.reshape(batch_size, -1, struct_vec.shape[-1])
        self.struct_attention(batch_size, max_sent_len, context_sentence_masks, struct_vec, None)

    def forward(self, context_pair_input_ids, context_pair_input_masks, type_ids, context_sentence_masks, labels,
                max_sent_len=None):
        batch_size = context_sentence_masks.shape[0]
        struct_vec = self.encoder(context_pair_input_ids, context_pair_input_masks)[0][:, 0, :]
        # distribute result into batch
        struct_vec = struct_vec.reshape(batch_size, -1, struct_vec.shape[-1])
        potentials, log_partition = self.struct_attention(batch_size, max_sent_len, context_sentence_masks, struct_vec,
                                                          labels)

        indices = torch.LongTensor(list(range(batch_size))).unsqueeze(1)
        potentials = potentials.masked_fill(potentials == float('-inf'), 0)
        single_tree_score = potentials[
            indices, labels[range(batch_size), 2, :], labels[range(batch_size), 0, :], labels[range(batch_size), 1,
                                                                                       :]]  # [bs, seq_len]
        single_tree_score = single_tree_score.sum(1)
        if (single_tree_score == float('-inf')).float().sum() == 1:
            print(single_tree_score)
            print(labels)
            print(potentials)
            exit()
        log_prob = single_tree_score - log_partition  # maximize this
        log_prob = log_prob.mean()  # sum of log prob equals joint prob in one batch

        return -log_prob
