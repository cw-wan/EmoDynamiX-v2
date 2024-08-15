import os
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from .model import Model

RELATION_DICT = {"Continuation": 0, "Question-answer_pair": 1, "Contrast": 2, "Q-Elab": 3, "Explanation": 4,
                 "Comment": 5, "Background": 6, "Result": 7, "Correction": 8, "Parallel": 9, "Alternation": 10,
                 "Conditional": 11, "Clarification_question": 12, "Acknowledgement": 13, "Elaboration": 14,
                 "Narration": 15, "Special": 16}


class StructuredDialogueDiscourseParser:
    def __init__(self,
                 ckpt_path,
                 max_contexts_length=48,
                 max_num_contexts=37,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.max_contexts_length = max_contexts_length
        self.max_num_contexts = max_num_contexts

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        encoder_config = AutoConfig.from_pretrained(os.path.join(ckpt_path, 'config.json'))
        encoder = AutoModel.from_config(encoder_config)
        self.model = Model(encoder_config, encoder=encoder, link_only=False)
        state_save_path = os.path.join(ckpt_path, 'pytorch_model.bin')
        print('Loading SDDP parameters from', state_save_path)
        self.model.load_state_dict(torch.load(state_save_path, map_location=torch.device('cpu')), strict=False)
        self.model.to(self.device)

    def _preprocess(self, batch):
        p_batch = []
        for sample in batch:
            p_sample = {}
            res = []
            for utt in sample:
                res.append(utt['speaker'] + ' , ' + utt['text'])
            p_sample['length'] = len(res)
            diff = self.max_num_contexts - len(res)
            res = res + ['dummy'] * diff
            p_sample['context'] = res
            p_sample['labels'] = [[], [], []]
            p_batch.append(p_sample)
        batch = p_batch
        contexts_pair_ids_batch, contexts_pair_ids_mask_batch, type_ids_batch, contexts_length_mask_batch, labels_batch = [], [], [], [], []
        for sample in batch:
            contexts, labels, length = sample['context'], sample['labels'], sample['length']
            first = []
            second = []
            for i in range(len(contexts)):
                for j in range(i, len(contexts)):
                    if i == j:
                        first.append("this is the placeholder for start of dialogue")  # can be any dummy sentence
                    else:
                        first.append(contexts[i])
                    second.append(contexts[j])

            tokenized_dict = self.tokenizer(first, second, padding='max_length', truncation='longest_first',
                                            max_length=self.max_contexts_length * 2)
            input_ids, attention_mask, type_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], \
                tokenized_dict['attention_mask']
            contexts_pair_ids_batch += input_ids
            contexts_pair_ids_mask_batch += attention_mask
            type_ids_batch += type_ids
            contexts_length_mask_batch.append(length)
            labels_batch.append(labels)

        type_ids_batch = torch.LongTensor(type_ids_batch)
        contexts_pair_ids_batch = torch.LongTensor(contexts_pair_ids_batch)
        contexts_pair_ids_mask_batch = torch.LongTensor(contexts_pair_ids_mask_batch)
        contexts_length_mask_batch = torch.LongTensor(contexts_length_mask_batch)
        labels_batch = torch.LongTensor(labels_batch)

        return contexts_pair_ids_batch, contexts_pair_ids_mask_batch, type_ids_batch, contexts_length_mask_batch, labels_batch

    def parse(self, batch):
        """
        Parse a batch of dialogues
        :param batch: List of dialogues, and every dialogue is a list of {'speaker': ..., 'text': ...}
        :return: Set of predicted dependency edges (triplets): {(x, y, r) ...}
        """
        self.model.eval()
        batch = self._preprocess(batch)
        input_ids = batch[0].numpy()
        str_keys = [" ".join(item) for item in input_ids.astype(str)]
        input_masks = batch[1].numpy().tolist()
        input_types = batch[2].numpy().tolist()
        mapping = {}
        for str_key, masks, types in zip(str_keys, input_masks, input_types):
            if str_key not in mapping:
                mapping[str_key] = [str_key, masks, types]
        with torch.no_grad():
            encoder_cache = self.model.encoder_inference(mapping)
            input_ids = batch[0].numpy()
            keys = [" ".join(item) for item in input_ids.astype(str)]
            struct_vec = [encoder_cache[key] for key in keys]
            struct_vec = torch.stack(struct_vec, 0)
            self.model.inference_forward(struct_vec.to(self.device), batch[3].to(self.device), self.max_num_contexts)
        tree_results = []
        relation_types = []
        for result in self.model.struct_attention.tree_results:
            tree_result, predicted_types = result
            tree_results += tree_result
            relation_types += predicted_types
        self.model.struct_attention.tree_results = []

        output = []
        for ds, r in zip(tree_results, relation_types):
            all_d = set()
            for d in ds:
                d = set([(dd, idx + 1, r[dd][idx + 1]) for idx, dd in enumerate(d[1:])])  # skip -1
                all_d.update(d)
            d = all_d
            output.append(d)
        return output
