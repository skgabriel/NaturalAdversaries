# NaturalAdversaries

## Training Data 

[DynaHate](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset)

[AdversarialNLI](https://github.com/facebookresearch/anli)

## Sampling 

Sampling with integrated gradients: `` python ./src/sampling/ig_sampling.py ``

Sampling with [Lime](https://arxiv.org/abs/1602.04938): `` python ./src/sampling/lime_sampling.py ``

## Prepared Data 

Train/dev/test splits for DynaHate can be found [here](https://drive.google.com/drive/folders/1uzQPeVwije2YlSXCTLUnNtYil2nSSrLV?usp=share_link).

Data splits for AdversarialNLI can be found [here](https://drive.google.com/drive/folders/1btUnoSUzaArh1qKNvCeOxxy1-bGbpNkj?usp=share_link).

## Training 

`` python ./src/modeling/finetune.py ``

## Generation

`` python ./src/modeling/generate.py ``

## Trained Models 

Trained adversarial generation models can be found here: https://huggingface.co/skg/na-models. 

Example Usage:

````

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
