import random
import logging

from itertools import chain
from copy import deepcopy

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences, truncate_sequences_dual
)

from .utils.dataset_walker import DatasetWalker

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<n_sep>", "<f_sep>"]
}
SPECIAL_TOKENS_LSTM = {
    "additional_special_tokens": ["<bos>", "<n_sep>", "<f_sep>", "<eod>", "<eos>"]
}
SPECIAL_TOKENS_VALUES = ["<n_sep>", "<f_sep>"]
SPECIAL_TOKENS_VALUES_LSTM = ["<bos>", "<n_sep>", "<f_sep>", "<eon>", "<eos>"]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type
        self.model_type = args.model_name_or_path.split("-")[0]
        if "/" in self.model_type:
            self.model_type = self.model_type.split("/")[-1]

        if self.model_type in ["bert", "roberta", "distilbert", "deberta"]:
            self.SPECIAL_TOKENS = SPECIAL_TOKENS
            self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
            self.narrative_sep = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS_VALUES[0])
            self.fact_sep = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS_VALUES[1])
        else:
            self.SPECIAL_TOKENS = SPECIAL_TOKENS_LSTM
            self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES_LSTM
            self.string_start = self.tokenizer.encode(self.SPECIAL_TOKENS_VALUES[0])
            self.narrative_sep = self.tokenizer.encode(self.SPECIAL_TOKENS_VALUES[1])
            self.fact_sep = self.tokenizer.encode(self.SPECIAL_TOKENS_VALUES[2])
            self.narrative_end = self.tokenizer.encode(self.SPECIAL_TOKENS_VALUES[3])
            self.string_end = self.tokenizer.encode(self.SPECIAL_TOKENS_VALUES[4])

        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)

        self.linking_samples = self._prepare_linking_samples()

        self._create_examples_linking()

    def _prepare_linking_samples(self):
        logger.info("Prepare the fact linking samples")
        linking_samples = []
        # only show progress bar in one process
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])):
            sample = {}
            sample["cid"] = log["cid"]
            sample["tid"] = log["tid"]
            if log.get("fid"):
                sample["fid"] = log["fid"]
            else:
                sample["fid"] = -1
            sample["text"] = log["text"]
            sample["label"] = label
            linking_samples.append(sample)
        return linking_samples

    def _create_examples_linking(self):
        logger.info("Creating examples")
        self.examples = []
        for sample in tqdm(self.linking_samples, disable=self.args.local_rank not in [-1, 0]):
            context_id = sample["cid"]
            turn_id = sample["tid"]
            fact_id = sample["fid"]
            label = sample["label"]
            text = sample["text"]
            if label is None:
                # create default target
                label = {"target": False, "linking": None}

            target = label["target"]

            context = []
            fact = []
            for line in text:
                if self.model_type in ["bert", "roberta", "distilbert", "deberta"]:
                    if line["type"] == "context":
                        context.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line["utter"])))
                    elif line["type"] == "fact":
                        fact.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line["utter"])))
                else:
                    if line["type"] == "context":
                        context.append(self.tokenizer.sentence_encode(line["utter"]))
                    elif line["type"] == "fact":
                        fact.append(self.tokenizer.sentence_encode(line["utter"]))

            # perform token-level truncation of history from the left
            no_trunc_token_num = 1 + len(context) + len(fact)  # [CLS], [SEP], <d_sep> and <s_sep>
            for node in fact:
                no_trunc_token_num += len(node)  # do not truncate statement

            if self.args.task_type == "nlu":
                truncated_context = truncate_sequences_dual(deepcopy(context), self.args.max_tokens-no_trunc_token_num)
            else:
                truncated_context = truncate_sequences(deepcopy(context), self.args.max_tokens - no_trunc_token_num)

            self.examples.append({
                "context": truncated_context,
                "fact": fact,
                "label": label,
                "target": target,
                "context_id": context_id,
                "turn_id": turn_id,
                "fact_id": fact_id
            })

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)


class FactLinkingDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(FactLinkingDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def build_input_from_segments(self, context, fact):
        """ Build a sequence of input from example """
        instance = {}

        sequence_context = deepcopy(context[0])
        for utter in context[1:]:
            sequence_context.append(self.narrative_sep)
            sequence_context += deepcopy(utter)

        sequence_fact = deepcopy(fact[0])
        for node in fact[1:]:
            sequence_fact.append(self.fact_sep)
            sequence_fact += deepcopy(node)

        sequence = [sequence_context, sequence_fact]

        if self.model_type in ["bert", "roberta", "distilbert", "deberta"]:
            instance["input_ids"] = self.tokenizer.build_inputs_with_special_tokens(sequence_context, sequence_fact)
            if self.model_type == "bert":
                instance["token_type_ids"] = self.tokenizer.create_token_type_ids_from_sequences(sequence_context, sequence_fact)
            else:
                instance["token_type_ids"] = None
        else:
            sequence_context.append(self.narrative_end)
            sequence_fact.append(self.string_end)
            instance["input_ids"] = [self.string_start] + sequence_context + sequence_fact
            instance["token_type_ids"] = None

        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence

    def __getitem__(self, index):
        example = deepcopy(self.examples[index])
        instance, _ = self.build_input_from_segments(example["context"], example["fact"])
        instance["label"] = example["target"]
        instance["context_id"] = example["context_id"]
        instance["turn_id"] = example["turn_id"]
        instance["fact_id"] = example["fact_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        mc_token_ids = [ins["mc_token_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {
            "context_ids": [ins["context_id"] for ins in batch],
            "turn_ids": [ins["turn_id"] for ins in batch],
            "fact_ids": [ins["fact_id"] for ins in batch]
        }

        if self.model_type in ["bert", "roberta", "distilbert", "deberta"]:
            pad_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        else:
            pad_token = self.tokenizer.encode(self.tokenizer.pad_token)

        input_ids = torch.tensor(pad_ids(input_ids, pad_token))
        if self.model_type == "bert":
            token_type_ids = torch.tensor(pad_ids(token_type_ids, pad_token))
        else:
            token_type_ids = torch.full_like(input_ids, 0)
        mc_token_ids = torch.tensor(mc_token_ids)
        lm_labels = torch.full_like(input_ids, 0)
        labels = torch.tensor(labels).long()

        return input_ids, token_type_ids, mc_token_ids, lm_labels, labels, data_info
