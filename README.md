## Forge-T5

### Forge-T5 is a transformer based large language model that is used to find a similarity score between two python code snippets. The model was fined tuned on [CodeT5](https://blog.salesforceairesearch.com/codet5/) model by Salesforce. Forge-T5 summarizes both the code snippets which are provided as an input, then this summarization is embedded into a sentence embedding using [Sentence BERT ](https://arxiv.org/abs/1908.10084). The cosine similarity of these two embedded vectors is calculated to then give out a similarity score. Check out the demo [here](https://huggingface.co/spaces/Paarth/ForgeT5).

![neural](https://user-images.githubusercontent.com/75850838/187042072-968fb2a5-0940-4c18-ba57-d8ed0927b1e3.jpg)
