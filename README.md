# LongPPL


## Requirements
Python 3.10 + Pytorch 2.3 + Transformers 4.45

```
pip install -r requirements.txt
```

## LongPPL
To calculate LongPPL, please run:
```
cd LongPPL
sh run_ppl.sh
```
The evaluation data can be downloaded from [GovReport (tokenized)](https://huggingface.co/datasets/emozilla/govreport-test-tokenized).

## LongCE
To conduct long-context finetuning with LongCE, run `accelerate config` and enable DeepSpeed acceleration. `deepspeed/zero3.json` was the configuration file used for training. 
```
cd finetune
sh train.sh
```
The training data can be downloaded from [PG19](https://huggingface.co/datasets/emozilla/pg19) and [Pile-arxiv](https://huggingface.co/datasets/suolyer/pile_arxiv).
To run models with eabf, please downgrade the version of `transformers` to `4.37.0`.

## Evaluation on Long-context Benchmark
In the paper, we evaluate models on [LongBench](https://github.com/THUDM/LongBench), [LongEval](https://github.com/DachengLi1/LongChat) and [RULER](https://github.com/nvtransfer/RULER). Please refer to the respective code repositories.
