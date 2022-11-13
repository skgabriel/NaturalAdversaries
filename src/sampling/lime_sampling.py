import argparse
from transformers import AutoTokenizer, pipeline
import torch.nn.functional as F
from lime_text import LimeTextExplainer
import csv
import torch
import numpy as np
import scipy
import tqdm
import json


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = pipeline("text-classification",model=args.model_dir, device=args.device, return_all_scores=True)

    def predictor(text):
        outputs = model(text)
        outputs = np.array([[s["score"] for s in x] for x in outputs])
        return outputs

    file_ = [row for row in csv.reader(open(args.input_file))]
    lime_scores = open(args.output_file,"w")
    if args.domain == "nli": 
       class_names = ["contradiction","neutral","entailment"]
    else:
       class_names = ["nothate","hate"]
    lime = LimeTextExplainer(class_names=class_names)
    batch_size=args.batch_size
    for i in tqdm.tqdm(range(0,len(file_),batch_size)):
        if args.domain == "nli":  
           try:
               token_ids = [tokenizer.encode(f[0] + " [SEP] " + f[1])[1:-1] for f in file_[i:i+batch_size]]
               exp = [lime.explain_instance(f[0] + " [SEP] " + f[1],predictor,num_features=20,num_samples=2000) for f in file_[i:i+batch_size]]
           except:
               continue
           pred = np.argmax(predictor([f[0] + " [SEP] " + f[1] for f in file_[i:i+batch_size]]),axis=1)
        else: 
           try:
               token_ids = [tokenizer.encode(f[0])[1:-1] for f in file_[i:i+batch_size]]
               exp = [lime.explain_instance(f[0],predictor,num_features=20,num_samples=2000) for f in file_[i:i+batch_size]]
           except:
               continue
           pred = np.argmax(predictor([f[0] for f in file_[i:i+batch_size]]),axis=1)
        k=[max(int(len(token_ids[o][1:-1])*args.keep_ratio),1) for o in range(len(exp))]
        exp_ = [sorted(zip([i[0] for i in exp[o].as_list()],[i[1] for i in exp[o].as_list()])) for o in range(len(exp))]
#        exp_ = [(exp_[j][0][-k[j]:],exp_[j][1][-k[j]:]) for j in range(len(exp_))]
#        exp_ = [torch.topk(exp[o].as_list(),k=int(len(token_ids[o][1:-1])*args.keep_ratio)) for o in range(len(exp))]
        for i,f in enumerate(file_[i:i+batch_size]):
            exp_ = exp[i].as_list()
            exp_t = [e[0] for e in exp_]
            exp_s = [e[1] for e in exp_]
            exp_s, exp_t = zip(*sorted(zip(exp_s, exp_t)))
            exp_s = exp_s[-k[i]:]
            exp_t = exp_t[-k[i]:]
            if args.domain == "nli":
               text = f[0] + " [SEP] " + f[1]
               label = f[2]
            else:
               text = f[0]
               label = f[1]   
            lime_scores.write(str(json.dumps({"text":text,"pred_label":int(pred[i]),"scores":exp_s,"keys":exp_t,"label":label,"split":args.split})) + "\n") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="lime_scores.jsonl")
    parser.add_argument("--input_file", type=str, default="nli_example.csv") #in format: example, label
    parser.add_argument("--model_dir", type=str, default="microsoft/deberta-base-mnli") #"Xuhui/ToxDect-roberta-large"
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=1) #-1 for cpu, >= 0 for cuda
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--keep_ratio", type=float, default=.2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--domain", type=str, default="nli") #hate"
    args = parser.parse_args()
