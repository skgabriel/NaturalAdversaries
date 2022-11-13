# Dataset Card

## Dataset Description

- **Repository:** https://github.com/skgabriel/NaturalAdversaries
- **Paper:** https://arxiv.org/abs/2211.04364
- **Point of Contact:** skgabrie@cs.washington.edu

### Dataset Summary

Adversarial examples generated using Integrated Gradient and LIME sampling

### Supported Tasks and Leaderboards

ANLI, DynaHate 

### Languages

English

## Dataset Structure

### Data Instances

150 instances for each domain 

### Data Fields

Hate Speech: text, label (from 3 annotators), method (of sampling)

NLI: premise, hypothesis, label (from 3 annotators), method (of sampling)

### Annotations

#### Who are the annotators?

US-based annotators from (https://arxiv.org/abs/1911.03891) and (https://arxiv.org/abs/2201.05955)

### Social Impact of Dataset

While there is a risk of any technologies aimed at mimicking natural language being used for malicious purposes, our work has wide-ranging potential societal benefit by improving fairness and real-world robustness of neural classifiers. Increasingly it has become clear that pretrained neural language models do not operate from a neutral perspective, and implicitly learn behaviors that pose real harm to users from training data (Jernite et al., 2022). We demonstrate that our framework is effective at generating adversaries that uncover model vulnerabilities for two well-studied domains (hate speech and NLI), and it is hypothetically extensible to other domains like automated fact-checking. Given the sensitive nature of toxic language and hate speech detection in particular, we strongly emphasize that the work is intended only for research purposes or improving robustness of automated systems. 

### Citation Information

@article{Gabriel2022NaturalAdversaries,
  title={NaturalAdversaries: Can Naturalistic Adversaries Be as Effective as Artificial Adversaries?},
  author={Saadia Gabriel and Hamid Palangi and Yejin Choi},
  journal={Findings of EMNLP},
  year={2022}
}
