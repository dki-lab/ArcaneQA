# ArcaneQA
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/dki-lab/ArcaneQA/issues)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)
[![paper](https://img.shields.io/badge/Paper-COLING2022-lightgrey?style=flat-square)](https://arxiv.org/pdf/2204.08109.pdf)

>ArcaneQA is a generation-based KBQA model built upon the encoder-decoder framework. It features two mutually-boosting modules: ***dynamic program induction*** and ***dynamic contextualized encoding*** in a unified framework. For dynamic program induction, we **unify the meaning representation in several KBQA datasets**, and our annotations are also available in this repo; For dynamic contextualized encoding, we **introduce a novel idea of applying pre-trained language models (PLMs) to KBQA**.

>This is the accompanying code&data for the paper "[ArcaneQA: Dynamic Program Induction and Contextualized Encoding for Knowledge Base Question Answering](https://arxiv.org/pdf/2204.08109.pdf)" published at COLING 2022.

<img width="677" alt="image" src="https://user-images.githubusercontent.com/15921425/193238484-399a05c3-50fa-44b0-add9-16a4ec1cb8fe.png">


## Package Description

This repository is structured as follows:

```
ArcaneQA/
├─ configs/
    ├─ train/: Configuration files for training
    ├─ infer/: Configuration files for inference
├─ data/: Processed data files for different KBQA datasets
├─ ontology/: Processed Freebase ontology files
    ├─ domain_dict: Mapping from a domain in Freebase Commons to all schema items in it
    ├─ domain_info: Mapping from a schema item to a Freebase Commons domain it belongs to
    ├─ fb_roles: Domain and range information for a relation (Note that here domain means a different thing from domains in Freebase Commons)
    ├─ fb_types: Class hierarchy in Freebase
    ├─ reverse_properties: Reverse properties in Freebase 
    ├─ commons/: Corresponding files of Freebase Commons subset
├─ el_results/: Entity linking results 
├─ vocabulary/: Preprocessed vocabulary
├─ cache/: Cached results for SPARQL queries, which are used to accelerate the experiments by caching many SPARQL query results offline
├─ saved_models/: Trained models
├─ source/:
    ├─ utils/:
        ├─ bert_interface.py: Interface to BERT 
        ├─ my_pretrained_transformer_indexer.py: Class for indexer for PLMs, adapted from AllenNLP
        ├─ my_pretrained_transformer_tokenizer.py: Class for tokenizer for PLMs, adapted from AllenNLP
        ├─ logic_form_util.py: Tools related to logical forms, including the exact match checker for two logical forms
        ├─ arcane_beam_search.py: Beam search for autoregressive decoding in ArcaneQA
        ├─ sparql_executor.py: Sparql-related tools
        ├─ kb_engine.py: Core functions for KB querying and constrained decoding
        ├─ sparql_cache.py: Cache executions of different types of Sparql queries
    ├─ run.py: Main function
    ├─ arcane.py: ArcaneQA model class
    ├─ arcane_reader.py: ArcaneQA dataset reader class
```


## Setup
We recommend you following the detailed instructions in https://github.com/dki-lab/GrailQA.

## Running Experiments
To train the model, run
```
$ PYTHONHASHSEED=23 python run.py 
train 
[your_config_file_for_training]  
--include-package 
arcane 
--include-package 
arcane_reader 
--include-package 
utils.bert_interface 
--include-package 
utils.my_pretrained_transformer_indexer 
--include-package 
utils.my_pretrained_transformer_tokenizer 
-s 
[your_path_specified_for_training]
```
To make predictions using a trained model, run
```
$ PYTHONHASHSEED=23 python run.py 
predict 
[path_for_trained_model] 
[path_for_test_dataset] 
--include-package 
arcane 
--include-package 
arcane_reader 
--include-package 
utils.bert_interface 
--include-package 
utils.my_pretrained_transformer_indexer 
--include-package 
utils.my_pretrained_transformer_tokenizer 
--use-dataset-reader 
--predictor 
seq2seq 
-c 
[your_config_file_for_inference]  
--output-file 
[output_file_name] 
--cuda-device 
0
```

## Citation
TBD
