import os
from numpy import sqrt
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
import random

class GPTDataSet(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./data",
        type_path="train",
        max_length = 150,
        pad_to_max_length = True,
        return_tensors = "pt"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.source_ids = []
        self.source_ids_loss = []
        self.source_mask = []
        self.source_mask_loss = []
        self.labels = []
        self.labels_loss = []
        data_path_source = os.path.join(data_dir, type_path + ".source")
        data_path_target = os.path.join(data_dir, type_path + ".target")

        data_path_loss = [row for row in csv.reader(open(os.path.join(data_dir, "loss.csv")))][1:]

        a = [l.strip() for l in open(data_path_source,'r', encoding='utf-8').readlines()]
        b = [l.strip() for l in open(data_path_target,'r', encoding='utf-8').readlines()]
        for i in range(len(a)):
            tokenization = tokenizer.encode_plus(
                a[i] + " " + b[i] + " " + tokenizer.eos_token, truncation = True, 
                max_length = max_length, pad_to_max_length=pad_to_max_length, 
                return_tensors=return_tensors)
            label = torch.clone(tokenization.input_ids)
            pad_inds = (label[0] == tokenizer.pad_token_id).nonzero(as_tuple=True)

            # Set the labels to -100, except for the actual eos token
            label[0][pad_inds[0][1:]] = -100

           # print(tokenization.input_ids, tokenization.attention_mask)
            self.source_ids.append(tokenization.input_ids)
            self.source_mask.append(tokenization.attention_mask)
            self.labels.append(label)


        for i in range(len(data_path_loss)):
            tokenization_loss = tokenizer.encode_plus(
                data_path_loss[i][0] + " " + tokenizer.eos_token, truncation = True, 
                max_length = max_length, pad_to_max_length=pad_to_max_length, 
                return_tensors=return_tensors)
            label_loss = torch.clone(tokenization_loss.input_ids)
            pad_inds_loss = (label_loss[0] == tokenizer.pad_token_id).nonzero(as_tuple=True)

            # Set the labels to -100, except for the actual eos token
            label_loss[0][pad_inds_loss[0][1:]] = -100

           # print(tokenization.input_ids, tokenization.attention_mask)
            self.source_ids_loss.append(tokenization_loss.input_ids)
            self.source_mask_loss.append(tokenization_loss.attention_mask)
            self.labels_loss.append(label_loss)            

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, index):
        loss_index = random.sample(range(len(self.source_ids_loss)),1)[0]
        return self.source_ids[index], self.source_mask[index], self.labels[index], self.source_ids_loss[loss_index], self.source_mask_loss[loss_index], self.labels_loss[loss_index]
