# NaturalAdversaries

## Training Data 

[DynaHate](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset)

[AdversarialNLI](https://github.com/facebookresearch/anli)

## Sampling 

Sampling with integrated gradients: `` python ./src/sampling/ig_sampling.py ``

Sampling with [Lime](https://arxiv.org/abs/1602.04938): `` python ./src/sampling/lime_sampling.py ``

## Generated Examples

[Hate speech and NLI examples](https://github.com/skgabriel/NaturalAdversaries/tree/main/data) (generated using either integrated gradients (ag-ig) or Lime (ag-lime))

## Training 

`` python ./src/modeling/finetune.py ``

## Generation

`` python ./src/modeling/generate.py ``

## Trained Models 

Trained adversarial generation models can be found here: https://huggingface.co/skg/na-models. 

Example Usage:

```

from transformers import GPT2Tokenizer, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained(model_dir)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

sequence = "[attr] br , ĠThey , Ġbought , Ġwent , Ġbut , Ġavailable , Ġmodel , Ġdeliberate , Ġwanted , > , ĠI , Ġdecided [label] 2 [text]" 

premise = "Grey<br>I went to the store to buy a new phone. The one I wanted was available. The salesperson showed me three different colors. I had a hard time choosing. I finally decided on the grey model. [SEP]"  

input_text = tokenizer(sequence + " " + premise,return_tensors="pt")

output_text = model.generate(**input_text,max_length=200,num_beams=5,repetition_penalty=2.5)

output_text = tokenizer.decode(output_text[0].tolist())

print(output_text.split("[SEP] ")[-1].replace("<|endoftext|>",""))

```

## Robustness Stress Tests

[SNLI-Hard](https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl)

[HateCheck](https://github.com/paul-rottger/hatecheck-data)

## Classifiers 

Links to tested classifiers can be found here:

Hate Speech:

[HateXplain model](https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain?text=I+like+you.+I+love+you)

[Roberta TwitterHate model](https://huggingface.co/Xuhui/ToxDect-roberta-large?text=I+like+you.+I+love+you)

NLI:

[DeBERTa MNLI model](https://huggingface.co/microsoft/deberta-base-mnli?text=%5BCLS%5D+I+love+you.+%5BSEP%5D+I+like+you.+%5BSEP%5D)

[QNLI model](https://huggingface.co/textattack/bert-base-uncased-QNLI?text=I+like+you.+I+love+you)
