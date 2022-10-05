# ArcaneQA
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/dki-lab/ArcaneQA/issues)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)
[![paper](https://img.shields.io/badge/Paper-COLING2022-lightgrey?style=flat-square)](https://arxiv.org/pdf/2204.08109.pdf)

>ArcaneQA is a generation-based KBQA model built upon the encoder-decoder framework. It features two mutually-boosting modules: ***dynamic program induction*** and ***dynamic contextualized encoding*** in a unified framework. For dynamic program induction, we **unify the meaning representation in several KBQA datasets**, and our annotations are also available in this repo; For dynamic contextualized encoding, we **introduce a novel idea of applying pre-trained language models (PLMs) to KBQA**.

>This is the accompanying code&data for the paper "[ArcaneQA: Dynamic Program Induction and Contextualized Encoding for Knowledge Base Question Answering](https://arxiv.org/pdf/2204.08109.pdf)" published at COLING 2022.

<img width="677" alt="image" src="https://user-images.githubusercontent.com/15921425/193238484-399a05c3-50fa-44b0-add9-16a4ec1cb8fe.png">

To train the model, run
```
$ PYTHONHASHSEED=23 python run.py 
train 
[your_config_file_for_training (e.g., grail_train.json)]  
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
[your_config_file_for_inference (e.g., grail_infer.json)]  
--output-file 
[output_file_name] 
--cuda-device 
0
```


We will provide more detailed instructions soon. Please stay tuned!
