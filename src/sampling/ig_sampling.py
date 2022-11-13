import json
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
import argparse 


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions[1:-1]

def pre_processing(obs, tokenizer, device):
    obs_tensor = torch.LongTensor([tokenizer.encode(obs)])
    if device == "cuda":
        obs_tensor.to('cuda')
    else:
        obs_tensor.to('cpu')
    return obs_tensor

def calculate_outputs_and_gradients(inputs, model, tokenizer, device="cpu", domain="nli"):

    #inner function for ig sampling
    def forward_func(inputs):
        output = model(inputs).logits
        return output.max(1).values

    # do the pre-processing
    predict_idx = None
    gradients = []
    outputs = []
    for input in inputs:
        with torch.no_grad():
             input = pre_processing(input, tokenizer, device)
             output = model(input.to(device)).logits
             outputs.append(torch.argmax(torch.nn.functional.softmax(output)))
             if args.domain == "nli":
                 lig = LayerIntegratedGradients(forward_func, model.deberta.embeddings)
             else:
                 lig = LayerIntegratedGradients(forward_func, model.roberta.embeddings)
             attributions, error_ = lig.attribute(inputs=input.to(device),baselines=torch.LongTensor([[0] * len(input)]).to(device),return_convergence_delta=True)
             gradients.append(attributions)
    return gradients, outputs

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    model.to(device)

    file_ = [row for row in csv.reader(open(args.input_file))]
    batch_size = args.batch_size
    ig_scores = open(args.output_file,"w")
    for i in tqdm.tqdm(range(0,len(file_),batch_size)):
        if args.domain == "nli":
            try:                                                                                                                                      
               token_ids = [tokenizer.encode(s[0] + " [SEP] " + s[1])[1:-1] for s in file_[i:i+batch_size]]                                                                  
            except:                                                                                                                                              
               continue                                                                                                                                           
            attr_,outputs =calculate_outputs_and_gradients([s[0] + " [SEP] " + s[1] for s in file_[i:i+batch_size]], model, tokenizer,device=device,domain=args.domain)     
        else:
            try:
               token_ids = [tokenizer.encode(s[0])[1:-1] for s in file_[i:i+batch_size]]
            except:
               continue
            attr_,outputs =calculate_outputs_and_gradients([s[0] for s in file_[i:i+batch_size]], model, tokenizer,device=device,domain=args.domain)
        attr_ = [summarize_attributions(attr_[o]) for o in range(len(attr_))]
        output_ = [torch.topk(attr_[o],k=max(1,int(len(token_ids[o])*args.keep_ratio))) for o in range(len(attr_))]
        for o in range(len(output_)):
            keys = list(set([tokenizer.convert_ids_to_tokens(token_ids[o][int(i)]) for i in output_[o][1].tolist()]))
            if args.domain == "nli":
                text = [s[0] + " [SEP] " + s[1] for s in file_[i:i+batch_size]][o]                                                                                                  
                label = [s[2] for s in file_[i:i+batch_size]][o] 
            else:
                text = [s[0] for s in file_[i:i+batch_size]][o]
                label = [s[1] for s in file_[i:i+batch_size]][o]
            split = [args.split for s in file_[i:i+batch_size]][o]
            import pdb; pdb.set_trace()
            line_ = {"text":text,"scores":[float(i) for i in output_[o].values],"keys":keys,"label":label,"split":split}
            ig_scores.write(str(json.dumps(line_)) +  "\n") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="ig_scores.jsonl")
    parser.add_argument("--input_file", type=str, default="nli_example.csv") #in format: example, label
    parser.add_argument("--model_dir", type=str, default="microsoft/deberta-base-mnli") #default="Xuhui/ToxDect-roberta-large")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--keep_ratio", type=float, default=.2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--domain", type=str, default="nli")
    args = parser.parse_args()
	
    main(args)
