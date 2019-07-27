# Second_Order SDP
Second Order Parser for Semantic Dependency Parsing

This repo contains the code forked from [Parser-v3](https://github.com/tdozat/Parser-v3) and used for the semantic dependency parser in Wang et al. (2019), [Second-Order Semantic Dependency Parsing with End-to-End Neural Networks](https://arxiv.org/abs/1906.07880). If you find our code is useful, please cite:
```
@inproceedings{wang-etal-2019-secodr,
    title = "{Second-Order Semantic Dependency Parsing with End-to-End Neural Networks}",
    author = {Wang, Xinyu and Huang, Jingxian and Tu, Kewei},
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
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
python3 -u main.py train GraphParserNetwork --config_file config_gen/sec_order.cfg 
```
This config file will run Mean Field Variational Inference for second order parts, and if you want to run with Loopy Belief Propagation, run
```bash
python3 -u main.py train GraphParserNetwork --config_file config_gen/sec_order_LBP.cfg 
```
