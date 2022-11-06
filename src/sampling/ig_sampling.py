import numpy as np
import torch
import tqdm
import torch.nn.functional as F
import torch.nn as nn
import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
from captum.attr import LayerIntegratedGradients
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import random

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions[1:-1]

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained('Xuhui/ToxDect-roberta-large')
model = AutoModelForSequenceClassification.from_pretrained('Xuhui/ToxDect-roberta-large')
model.eval()
model.to(device)

def forward_func(inputs):
    output = model(inputs).logits
    return output.max(1).values

def pre_processing(obs, tokenizer, device):
    obs_tensor = torch.LongTensor([tokenizer.encode(obs)])
    if device == "cuda":
        obs_tensor.to('cuda')
    else:
        obs_tensor.to('cpu')
    return obs_tensor

def calculate_outputs_and_gradients(inputs, model, tokenizer, device="cpu"):
    # do the pre-processing
    predict_idx = None
    gradients = []
    outputs = []
    for input in inputs:
        with torch.no_grad():
             input = pre_processing(input, tokenizer, device)
             output = model(input.to(device)).logits
             outputs.append(torch.argmax(torch.nn.functional.softmax(output)))
             lig = LayerIntegratedGradients(forward_func, model.roberta.embeddings)
             attributions, error_ = lig.attribute(inputs=input.to(device),baselines=torch.LongTensor([[0] * len(input)]).to(device),return_convergence_delta=True)
             gradients.append(attributions)
    return gradients, outputs

file_ = [row for row in csv.reader(open("/home/saadiag/lime/data/test_dyna.csv"))]

batch_size = 1
ig_scores = csv.writer(open("dyna_ig_test_scores.csv","w"))
for i in tqdm.tqdm(range(0,len(file_),batch_size)):
    token_ids = [tokenizer.encode(s[0])[1:-1] for s in file_[i:i+batch_size]]
    attr_,outputs =calculate_outputs_and_gradients([s[0] for s in file_[i:i+batch_size]], model, tokenizer,device=device)
    attr_ = [summarize_attributions(attr_[o]) for o in range(len(attr_))]
    output_ = [torch.topk(attr_[o],k=int(len(token_ids[o])*.2)) for o in range(len(attr_))]
    for o in range(len(output_)):
        keys = list(set([tokenizer.convert_ids_to_tokens(token_ids[o][int(i)]) for i in output_[o][1].tolist()]))
        text = [s[0] for s in file_[i:i+batch_size]][o]
        label = [s[1] for s in file_[i:i+batch_size]][o]
        split = ["test" for s in file_[i:i+batch_size]][o]
        ig_scores.writerow([text,output_[o],keys,label,split]) #,[s[header.index("split")] for s in file_[i:i+batch_size]][o]])
