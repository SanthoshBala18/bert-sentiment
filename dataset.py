import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# The bigger, the better. Depends on your GPU capabilities.
BATCH_SIZE = 16
MAX_SENTENCE_LENGTH = 256


class SentimentDataset:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def _rpad(array, n):
        current_len = len(array)
        if current_len > n:
            return array[:n]
        extra = n - current_len
        return array + ([0] * extra)

    def convert_to_embedding(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = tokens[:MAX_SENTENCE_LENGTH - 2]
        bert_sent = self._rpad(self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"]),
                               n=MAX_SENTENCE_LENGTH)

        return bert_sent

    def convert_data_to_embeddings(self, sentences_with_labels):
        for i in range(len(sentences_with_labels)):
            sentence = sentences_with_labels.iloc[i,0]
            label = sentences_with_labels.iloc[i,1]
            bert_sent = self.convert_to_embedding(sentence)
            yield torch.tensor(bert_sent), torch.tensor(label, dtype=torch.int64)

    @staticmethod
    def _parse_imdb_line(line):
        line = line.strip().lower()
        line = line.replace("&nbsp;", " ")
        line = re.sub(r'<br(\s\/)?>', ' ', line)
        line = re.sub(r' +', ' ', line)  # merge multiple spaces into one

        return line

    @staticmethod
    def _read_imdb_data(filename):
        data = []
        for line in open(filename, 'r', encoding="utf-8"):
            data.append(SentimentDataset._parse_imdb_line(line))

        return data

    def prepare_dataloader_from_examples(self, data, sampler=None):
        dataset = list(self.convert_data_to_embeddings(data))

        sampler_func = sampler(dataset) if sampler is not None else None
        dataloader = DataLoader(dataset, sampler=sampler_func, batch_size=BATCH_SIZE)

        return dataloader

    def prepare_dataloader(self, filename, sampler=None):
        data = pd.read_csv(filename)

        return self.prepare_dataloader_from_examples(data, sampler=sampler)

    def prepare_test_dataloader(self, filename):
        data = pd.read_csv(filename)
        data['sentiment'] = [0]*len(data)

        return self.prepare_dataloader_from_examples(data, sampler=sampler)


