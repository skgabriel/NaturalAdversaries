import csv
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import math
import sys
from pathlib import Path
from sklearn.utils import shuffle
from utils import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
from transformers import AdamW, get_linear_schedule_with_warmup
import tqdm 
import ast
import transformers

random.seed(0)

def convert_list(s):
    print(s)
    s1 = s.split("', ")[0].replace("[('","")
    try:
        s2 = s.split("', ")[1].split("',")[0].replace("')]","")
    except:
        import pdb; pdb.set_trace()
    return [(s1,s2)]

def flip_label(s):
    if "label] 2" in s:
       s = s.replace("label] 2","label] 0")
    elif "label] 0" in s:
       s = s.replace("label] 0","label] 2")
    elif "label] 1" in s:
       s = s.replace("label] 1","label] 0")
    return s

def get_label(s):
    return int(s.split("label] ")[1][0])

def check_length(s_,max_length=500):
    if len(s_["input_ids"][0]) > max_length:
       s_["input_ids"] = torch.tensor([s_["input_ids"][0][:-3][:max_length-4].tolist() + s_["input_ids"][0][-3:].tolist()])
       s_["attention_mask"] = torch.tensor([s_["attention_mask"][0][:-3][:max_length-4].tolist() + s_["attention_mask"][0][-3:].tolist()])
    return s_

def generate_inferences(
    data_path, data_type, model, model_d, tokenizer, tokenizer_d, device, model_type, gen_type = "beam", num_beams = 5,
    max_length = 1000, pad_to_max_length = True, return_tensors = "pt",
    save_dir = "inferences/", fname = "predict.csv", use_loss=False
    ):

    data_path_source = os.path.join(data_path, data_type + ".source")
    data_path_target = os.path.join(data_path, data_type + ".target")
    src = open(data_path_source,'r', encoding='utf-8').readlines()
    src = [flip_label(s[:-1]) for s in src]
    tgt = open(data_path_target,'r', encoding='utf-8').readlines()
    tgt = [s.strip() for s in tgt]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    task = "anli"
    already = []
    with open(os.path.join(save_dir,data_type + "_" + gen_type + "_" + fname), "w") as f:
        f = csv.writer(f)
        f.writerow(["text","label","model_label","orglabel","method"])
        for t in tqdm.tqdm(range(len(src))):
            #print(src[t] + " [SEP] " + tgt[t].split(" [SEP]")[0])
            src_ = tokenizer.encode_plus(src[t]  + " " + tgt[t].split(" [SEP]")[0] + " [SEP] ",return_tensors=return_tensors) #" [attr] " + " , ".join(ast.literal_eval(data[t][3])) + " [label] " + str(flip_label(int(data[t][1]))) + " [text] " + data[t][0].split("[SEP]")[0] + "[SEP]",return_tensors=return_tensors)
            if gen_type == "beam":
                generated_ids = model.generate(
                    input_ids=src_.input_ids.to(device), attention_mask=src_.attention_mask.to(device), num_beams=num_beams, 
                    max_length=max_length, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True, use_cache=True,)
            elif gen_type == "topk":
                generated_ids = model.generate(
                    input_ids=src_.input_ids.to(device), attention_mask=src_.attention_mask.to(device),
                    do_sample=True, top_k=num_beams, max_length=max_length, repetition_penalty=2.5,length_penalty=1.0, 
                    early_stopping=True, use_cache=True,)
            output = tokenizer.decode(generated_ids[0]).split("[text] ")[1].replace(" <|endoftext|>","")
            f.writerow([output,str(get_label(flip_label(src[t]))),0,get_label(src[t]),"ag-lime"])
            
def get_dataloader(tokenizer, out_, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
    dataset = GPTDataSet(tokenizer, data_dir=out_, type_path=type_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def is_loss_decreasing(old_loss, new_loss, thresh_percent = 0.01):
    if old_loss is None:
        return True
    if new_loss > old_loss:
        return False
    if old_loss == 0:
        return False
    
    diff = (old_loss - new_loss)/(old_loss)
    print("percent diff:", diff)
    return diff > thresh_percent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_loss', action="store_const",const=True)
    parser.add_argument('--attr_type',type=str,default="lime")
    parser.add_argument('--dataset', type=str, default='anli')
    parser.add_argument('--parallelize',type=bool,default=True)
    parser.add_argument('--model_dir', type=str, default='models/')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--n_batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--grad_accum_steps', type=int, default = 1)
    parser.add_argument('--warmup_steps', type=int, default = 1e2)
    parser.add_argument('--model_type', type=str, default = "gpt2")
    args = parser.parse_args()
    print(args)

    # Make easier var names
    batch_size = args.n_batch
    epochs = args.n_iter
    gradient_accumulation_steps = args.grad_accum_steps
    warmup_steps = args.warmup_steps
    model_type= args.model_type
    save_dir = os.path.join("./", args.dataset + "_" + args.model_dir.replace("/","") + "_" + args.attr_type)

    dims = ["keys","label","text"]

    out_ = "../" + args.dataset + "/" + args.attr_type

    dims_tokens = ["[" + x + "]" for x in dims]

    # Use the GPT2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)

    tokenizer.add_tokens(dims_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    n_vocab = len(tokenizer)
    print("Encoding dataset...", n_vocab)

    # Training dataloader
    train_dataloader = get_dataloader(tokenizer, out_, "train", batch_size, True)
    val_dataloader = get_dataloader(tokenizer, out_, "train", batch_size, True)
    if args.use_loss:
       model_pt = [d for d in os.listdir("../" + args.model_type + "/" + args.dataset + "_loss_" + args.model_dir.replace("/","") + "_0.0002_" + args.attr_type) if "epoch" in d]
    else:
       model_pt = [d for d in os.listdir("../" + args.model_type + "/" + args.dataset + "_" + args.model_dir.replace("/","") + "_0.0002_" + args.attr_type) if "epoch" in d]
    model_pt = sorted(model_pt)[-2]
    if args.use_loss:
       model = GPT2LMHeadModel.from_pretrained("../" + args.model_type + "/" + args.dataset + "_loss_" + args.model_dir.replace("/","") + "_0.0002_" + args.attr_type + "/" + model_pt) #epoch_4, epoch_6
    else:
       model = GPT2LMHeadModel.from_pretrained("../" + args.model_type + "/" + args.dataset + "_" + args.model_dir.replace("/","") + "_0.0002_" + args.attr_type + "/" + model_pt)
    model.resize_token_embeddings(len(tokenizer))

    if args.parallelize:
       model.parallelize()
    model = model.to(device)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr = args.lr,)

    t_total = (
        (len(train_dataloader.dataset) // (batch_size * max(1, 1)))
        // gradient_accumulation_steps
        * epochs
    )

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = t_total)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
   
    file_name = args.dataset + "_" + args.model_type + "_" + args.attr_type + "_infs"
    if args.use_loss:
       file_name += "_use_loss"
       generate_inferences(out_, "test", model, None, tokenizer, None, device, model_type, gen_type = "topk",save_dir=file_name,use_loss=True)
    else:
       generate_inferences(out_, "test", model, None, tokenizer, None, device, model_type, gen_type = "beam",save_dir=file_name,use_loss=False)
