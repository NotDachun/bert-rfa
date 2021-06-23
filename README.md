# Repurposing Pre-trained Language Models with Drop-in Attention Mechanisms

Language models based on transformers currently attain state-of-the-art results across a variety of tasks with attention being the mechanism behind recent improvements. Traditional softmax based attention mechanisms, however, scale quadratically in relation to the input sequence length, therefore preventing the efficient processing of long context strings. At Noah's Ark, we have built several attention variants that improve the inference time and space efficiency, such as:
- [Random Feature Attention](https://arxiv.org/abs/2103.02143)

This demonstration shows an easy way in practice to convert pre-trained models from the HuggingFace community to use a drop-in attention variant, allowing for a simple workflow to  create a more efficient alternative with similar performance to the baseline.

## Set Up
Pull code and install environment
```
git clone https://github.com/NotDachun/huggingface-ark.git
cd huggingface-ark
pip install -r requirements.txt
```

## Pretraining Models
This [notebook](https://github.com/NotDachun/huggingface-ark/blob/main/convert_model_to_RFA.ipynb) demonstrates our procedure for training RoBERTa with RFA starting from the RoBERTa checkpoint. The same procedure can be followed to get a version of other existing pretrained models.

## MNLI
Training script adapted from HuggingFace [transformers](https://github.com/huggingface/transformers) library: `run_mnli.py`. Check out a simple example training [script](https://github.com/NotDachun/huggingface-ark/blob/main/train_roberta_wiki2.sh).

## Future development
- Add the RFA code into the repository as a module to remove the dependency on rfa repository (which requires a user to set up the entire fairseq library)
- Do the same for the linfa module and add an interface to HuggingFace
- Modularize the interface to HuggingFace (can reference the convert_model_to_rfa notebook)
- Add the ability to use drop-in replacements for causal language models
- Make the repository a pip installable package (write a setup.py)

## Further experimentation
- Efficiency of training models from scratch with attention variants
- Performance on longer sequence lengths with attention variants


