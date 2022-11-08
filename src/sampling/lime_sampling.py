from transformers import RobertaForSequenceClassification, RobertaTokenizer, AutoTokenizer
import transformers
import torch.nn.functional as F
from lime_text import LimeTextExplainer
import csv
import torch
import numpy as np
import shap
import scipy
import tqdm
import json

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base-mnli")
model = transformers.pipeline("text-classification",model='microsoft/deberta-base-mnli', device=7, return_all_scores=True)

def predictor(text):
    outputs = model(text)
    outputs = np.array([[s["score"] for s in x] for x in outputs])
    return outputs

file_ = [json.loads(row) for row in open("../data/anli_v1.0/R3/train.jsonl").readlines()]
header = file_[0]
#file_ = [f for f in file_[1:] if "4" in f[header.index("round")]]
print(len(file_))
file_ = file_[40000:42000]
lime_scores = csv.writer(open("anli_train_scores15.csv","w"))
lime_scores.writerow(["text","pred_label","lime_scores","shap_scores","tokens","gold_label","split"])
class_names = ["contradiction","entailment","neutral"]
lime = LimeTextExplainer(class_names=class_names)
shap = shap.Explainer(model)
batch_size=1

for i in tqdm.tqdm(range(0,len(file_),batch_size)):
    exp = [lime.explain_instance(f["context"] + " [SEP] " + f["hypothesis"],predictor,num_features=20,num_samples=2000) for f in file_[i:i+batch_size]]
    exp_shap = shap([f["context"] + " [SEP] " + f["hypothesis"] for f in file_[i:i+batch_size]])
    pred = np.argmax(predictor([f["context"] + " [SEP] " + f["hypothesis"] for f in file_[i:i+batch_size]]),axis=1) #int(np.argmax(F.softmax(model(input_ids=input["input_ids"],attention_mask=input["attention_mask"])[0][0]).detach().numpy()))
    for i,f in enumerate(file_[i:i+batch_size]):
        lime_scores.writerow([f["context"] + " [SEP] " + f["hypothesis"],pred[i],exp[i].as_list(),exp_shap[i].values.tolist(),exp_shap[i].data.tolist(),f["label"],"train"])
