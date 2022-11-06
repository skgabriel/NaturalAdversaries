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

def generate_inferences(
    data_path, data_type, model, tokenizer, device, model_type, gen_type = "beam", num_beams = 3,
    max_length = 500, pad_to_max_length = True, return_tensors = "pt",
    save_dir = "inferences/", fname = "predict.txt"
    ):

    data_path_source = os.path.join(data_path, data_type + ".source")
    src = open(data_path_source,'r', encoding='utf-8').readlines()
    src = [s[:-1] for s in src]

    tokenizations = [tokenizer.encode_plus(s, return_tensors=return_tensors) for s in src]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, model_type)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir,data_type + "_" + gen_type + "_" + fname), "w") as f:
        for t in tokenizations:
            if gen_type == "beam":
                generated_ids = model.generate(
                    input_ids=t.input_ids.to(device), attention_mask=t.attention_mask.to(device), num_beams=num_beams, 
                    max_length=max_length, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True, use_cache=True,)
            elif gen_type == "topk":
                generated_ids = model.generate(
                    input_ids=t.input_ids.to(device), attention_mask=t.attention_mask.to(device),
                    do_sample=True, top_k=num_beams, max_length=max_length, repetition_penalty=2.5,length_penalty=1.0, 
                    early_stopping=True, use_cache=True,)
            output = tokenizer.decode(generated_ids[0])
            f.write(output + "\n")

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
lrs = [2e-6,2e-5,2e-4,2e-3]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallelize',type=bool,default=True)
    parser.add_argument('--dataset', type=str, default='dyna')
    parser.add_argument('--attr_type',type=str,default="lime")
    parser.add_argument('--model_dir', type=str, default='models_')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--n_batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
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
    args.model_dir = args.model_dir + str(args.lr)
    save_dir = os.path.join("../", args.model_type, args.dataset + "_" + args.model_dir + "_" + args.attr_type + "/")

    dims = ["attr","label","text"]

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
    val_dataloader = get_dataloader(tokenizer, out_, "dev", batch_size, True)

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.resize_token_embeddings(len(tokenizer))

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
    
    train_loss = []
    val_loss = []

    old_val_loss = None
    avg_val_loss = None
    val_loss_decreasing = True
    i = 0

    # Train the model
    while i < epochs and val_loss_decreasing:
        save_path = os.path.join(save_dir, "epoch_" + str(i))
        print(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        model.train()
        cur_train_loss = 0.0
        print("running epoch", i)
        train_count = 0
        for batch_input, batch_mask, batch_labels, batch_input_loss, batch_mask_loss, batch_labels_loss in tqdm.tqdm(train_dataloader):
            # print(batch_input[0], batch_mask[0], batch_labels[0])
            if train_count % 1000 == 0:
                print("count", train_count)
            outputs = model(batch_input.to(device),labels=batch_labels.to(device),attention_mask=batch_mask.to(device),token_type_ids=None)
            loss, logits = outputs[:2]
            cur_train_loss += loss.detach().to('cpu')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()    
            scheduler.step()
            train_count += 1
        
        old_val_loss = avg_val_loss
        cur_val_loss = 0.0
        
        model.eval()
        val_count = 0
        with torch.no_grad():
            for batch_input, batch_mask, batch_labels, batch_inputs_loss, batch_mask_loss, batch_labels_loss in tqdm.tqdm(val_dataloader):
                val_outputs = model(batch_input.to(device),labels=batch_labels.to(device),attention_mask=batch_mask.to(device),token_type_ids=None)

                temp_val_loss, temp_val_logits = val_outputs[:2]
                cur_val_loss += temp_val_loss.detach().to('cpu')

                val_count += 1

        # compute averages
        avg_train_loss = cur_train_loss/train_count

        train_loss.append(avg_train_loss)
        
        avg_val_loss = cur_val_loss/val_count
        val_loss.append(avg_val_loss)

        val_loss_decreasing = is_loss_decreasing(old_val_loss, avg_val_loss)
        print(old_val_loss, avg_val_loss, val_loss_decreasing)

        model.save_pretrained(save_path)
        print("train loss", avg_train_loss)
        print("val loss:", avg_val_loss)

        i += 1

    with open(os.path.join(save_dir, "train_loss_" + str(args.lr) + ".txt"), "w") as f:
        for t in train_loss:
            f.write(str(t.item()))
            f.write("\n")
    with open(os.path.join(save_dir, "val_loss_" + str(args.lr) + ".txt"), "w") as f:
        for t in val_loss:
            f.write(str(t.item()))
            f.write("\n")
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000"))
    generate_inferences(out_, "test", model, tokenizer, device, model_type, gen_type = "beam",save_dir=args.dataset + "_" + args.model_type + "_infs")
    generate_inferences(out_, "test", model, tokenizer, device, model_type, gen_type = "topk",save_dir=args.dataset + "_" + args.model_type + "_infs")
