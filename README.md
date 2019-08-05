# Second_Order SDP
Second Order Parser for Semantic Dependency Parsing

This repo contains the code forked from [Parser-v3](https://github.com/tdozat/Parser-v3) and used for the semantic dependency parser in Wang et al. (2019), [Second-Order Semantic Dependency Parsing with End-to-End Neural Networks](https://arxiv.org/abs/1906.07880). If you find our code is useful, please cite:
```
@inproceedings{wang-etal-2019-second,
    title = "Second-Order Semantic Dependency Parsing with End-to-End Neural Networks",
    author = "Wang, Xinyu  and
      Huang, Jingxian  and
      Tu, Kewei",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics"
}
```

## How to use
### Training
Our second order parser can be trained by simply running
```bash
python3 -u main.py train GraphParserNetwork --config_file config/sec_order.cfg 
```
This config file will run Mean Field Variational Inference for second order parts, and if you want to run with Loopy Belief Propagation, run
```bash
python3 -u main.py train GraphParserNetwork --config_file config/sec_order_LBP.cfg 
```
### Parsing
A trained model can be run by calling
```bash
python3 main.py --save_dir $SAVEDIR run $DATADIR --output_dir results 
```
The parsed result will be saved `results/` directory. The `$SAVEDIR` is the directory of the model, for example, if you trained `config/sec_order.cfg`, the model will be saved in `saves/SemEval15/DM/MF_dm_3iter`. The `$DATADIR` is the directory of the data in `CONLLU` format.

## OOM issue
To avoid out of memory in training phase, our parser should be trained with 12GB gpu memory, and no longer than 60 words for each sentence. The number of iterations for mean field variational inference is at most 3 and at most 2 for loopy belief propagation in a 12GB Titan X gpu. If you have a larger gpu, such as Tesla P40 24GB, loopy belief propation can be also trained with 3 iterations. To set the number of iterations, set `num_iteration` in `SecondOrderGraphIndexVocab` or `SecondOrderGraphLBPVocab` of the config file. Another way is reduce the training `batch_size` in `CoNLLUTrainset` of the config file.

## Details
If you want to see some details of our parser, the source code for our parser is in `parser/structs/vocabs/second_order_vocab.py` for Mean Field Variational Inference and `second_order_LBP_vocab.py` for Loopy Belief Propagation in the same directory.

##Contact
If you have any questions, feel free to contact with me with [email](mailto:wangxy1@shanghaitech.edu.cn)