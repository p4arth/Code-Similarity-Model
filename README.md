## Forge-T5

### Forge-T5 is a transformer based large language model that is used to find a similarity score between two python code snippets. The model was fined tuned on [CodeT5](https://blog.salesforceairesearch.com/codet5/) model by Salesforce. Forge-T5 summarizes both the code snippets which are provided as an input, then this summarization is embedded into a sentence embedding using [Sentence BERT ](https://arxiv.org/abs/1908.10084). The cosine similarity of these two embedded vectors is calculated to then give out a similarity score. Check out the demo [here](https://huggingface.co/spaces/Paarth/ForgeT5).

![neural](C:\Users\PAARTH\Desktop\neural.jpg)