import torch
import torch.nn as nn
import torch_struct
from .draw_tree import draw_tree
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
import torch.nn.functional as F


class Struct_Attention(nn.Module):
    def __init__(self, config, link_only):
        super(Struct_Attention, self).__init__()
        self.tree_results = []  # for storing induced trees
        self.rnn_f = nn.LSTM(config.hidden_size, config.hidden_size, bidirectional=True, batch_first=True)
        self.rnn_b = nn.LSTM(config.hidden_size, config.hidden_size, bidirectional=True, batch_first=True)
        if link_only:
            self.num_types = 1
        else:
            self.num_types = 17
        self.link_only = link_only
        self.transform_list = nn.ModuleList([nn.Linear(config.hidden_size * 2, self.num_types)])

    def expand_matrix(self, matrix):
        seq_len = matrix.shape[2]
        matrix = matrix.reshape(-1, seq_len, seq_len)
        batch_size = matrix.shape[0]
        matrix_expand = torch.cat((torch.diagonal(matrix, dim1=-2, dim2=-1).unsqueeze(1), matrix), 1)
        pad = torch.tensor([float('-inf')] * (1 + seq_len), device=matrix_expand.device).unsqueeze(0).expand(batch_size,
                                                                                                             seq_len + 1).unsqueeze(
            2)
        matrix_expand = torch.cat((pad, matrix_expand), 2)
        mask = torch.eye(matrix_expand.shape[1], device=matrix_expand.device)
        matrix_expand = matrix_expand.masked_fill(mask == 1, 0)
        matrix_expand = matrix_expand.reshape(-1, self.num_types, seq_len + 1, seq_len + 1)

        return matrix_expand

    def apply_mask(self, potentials, context_sentence_masks):
        seq_len = potentials.shape[2]

        # only the dummy root relation can be Special (the last one)
        if not self.link_only:
            potentials[:, :-1, range(seq_len), range(seq_len)] = float("-inf")
            indices = torch.triu_indices(seq_len, seq_len, offset=1)
            potentials[:, -1, indices[0], indices[1]] = float("-inf")

        potentials = potentials.reshape(-1, seq_len, seq_len)
        context_sentence_masks = torch.repeat_interleave(context_sentence_masks, repeats=self.num_types)
        # construct length mask
        length_mask = [[[0] * length + [1] * (seq_len - length)] * seq_len for length in context_sentence_masks]
        length_mask = torch.BoolTensor(length_mask).to(potentials.device)
        potentials = potentials.masked_fill(length_mask, float('-inf'))

        # we only want the upper right part of the potential matrix
        mask = torch.triu(torch.ones_like(potentials))
        potentials = potentials.masked_fill(mask == 0, float('-inf'))

        potentials = potentials.reshape(-1, self.num_types, seq_len, seq_len)

        return potentials

    def contextualize(self, potentials_vec, context_sentence_masks):
        batch_size = potentials_vec.shape[0]
        seq_len = potentials_vec.shape[1]
        dim = potentials_vec.shape[3]

        # reshape potentials for rnn input
        rows = potentials_vec.reshape(-1, seq_len, dim)
        rows_f = torch.flip(rows, [1])
        rows_f = pack_padded_sequence(rows_f, list(range(1, seq_len + 1))[::-1] * batch_size, batch_first=True,
                                      enforce_sorted=False)
        rows_f = self.rnn_b(rows_f)[0]
        rows_f = pad_packed_sequence(rows_f, batch_first=True, total_length=seq_len)[0]
        rows_f = torch.flip(rows_f, [1])
        rows_f = rows_f.reshape(batch_size, seq_len, seq_len, 2, dim)
        rows_f = rows_f[:, :, :, 1, :]

        rows_b = rows  # for notation consistency
        rows_b = pack_padded_sequence(rows_b, [i.data.cpu().numpy().tolist() for i in context_sentence_masks for _ in
                                               range(seq_len)], batch_first=True, enforce_sorted=False)
        rows_b = self.rnn_f(rows_b)[0]
        rows_b = pad_packed_sequence(rows_b, batch_first=True, total_length=seq_len)[0]
        rows_b = rows_b.reshape(batch_size, seq_len, seq_len, 2, dim)
        rows_b = rows_b[:, :, :, 1, :]
        rows = torch.cat([rows_f, rows_b], -1)

        # another direction
        cols = potentials_vec.transpose(2, 1).reshape(-1, seq_len, dim)
        cols_b = cols
        cols_b = pack_padded_sequence(cols_b, list(range(1, seq_len + 1)) * batch_size, batch_first=True,
                                      enforce_sorted=False)
        cols_b = self.rnn_b(cols_b)[0]
        cols_b = pad_packed_sequence(cols_b, batch_first=True, total_length=seq_len)[0]
        cols_b = cols_b.reshape(batch_size, seq_len, seq_len, 2, dim)
        cols_b = cols_b[:, :, :, 1, :]

        cols_f = torch.flip(cols, [1])
        cols_f = pack_padded_sequence(cols_f, [seq_len for i in range(seq_len)][::-1] * batch_size, batch_first=True,
                                      enforce_sorted=False)
        cols_f = self.rnn_f(cols_f)[0]
        cols_f = pad_packed_sequence(cols_f, batch_first=True, total_length=seq_len)[0]
        cols_f = torch.flip(cols_f, [1])
        cols_f = cols_f.reshape(batch_size, seq_len, seq_len, 2, dim)
        cols_f = cols_f[:, :, :, 1, :]
        cols = torch.cat([cols_f, cols_b], -1)
        cols = cols.transpose(2, 1)
        # merge
        potentials_vec = rows + cols

        return potentials_vec

    def forward(self, batch_size, seq_len, context_sentence_masks, struct_vec, input_labels):
        dim = struct_vec.shape[-1]
        tri_idx = torch.triu_indices(seq_len, seq_len).numpy()
        potentials_vec = torch.zeros((batch_size, seq_len, seq_len, dim), device=struct_vec.device)
        potentials_vec[:, tri_idx[0], tri_idx[1]] = struct_vec
        potentials_vec = self.contextualize(potentials_vec, context_sentence_masks)
        potential_types = self.transform_list[0](potentials_vec).permute(0, 3, 1, 2)
        potential_types = self.apply_mask(potential_types, context_sentence_masks)
        labeled_potentials = torch.logsumexp(potential_types, 1)
        log_partition = torch_struct.NonProjectiveDependencyCRF(labeled_potentials, multiroot=True,
                                                                lengths=context_sentence_masks).partition
        potentials_expand = self.expand_matrix(potential_types)

        if input_labels is None:
            # single root inference. we give up on dangling edus
            potential_types = potential_types + torch.diag_embed(
                torch.tensor([0.] + [float('-inf')] * (seq_len - 1)).to(potential_types.device), offset=0, dim1=-2,
                dim2=-1)
            # i, j means there is an arc from j to i
            score_matrix = potential_types.max(1)[0].transpose(2, 1).clone().detach()
            score_matrix[score_matrix == float('-inf')] = np.nan
            # expand the potential matrix
            score_matrix = torch.cat((torch.diagonal(score_matrix, dim1=-2, dim2=-1).unsqueeze(2), score_matrix), 2)
            pad = torch.tensor([np.nan] * (1 + seq_len), device=score_matrix.device).unsqueeze(0).unsqueeze(0).expand(
                batch_size, 1, seq_len + 1)
            score_matrix = torch.cat((pad, score_matrix), 1)
            tree_result = draw_tree(score_matrix.data.cpu().numpy(), context_sentence_masks.clone().detach())
            predicted_types = potentials_expand.argmax(1)
            self.tree_results.append((tree_result, predicted_types.data.cpu().numpy().tolist()))

        return potentials_expand, log_partition
