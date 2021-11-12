# Out_of_domain_ABSA

First of all, you need to load your data in "data" folder. You can see the data format with my preprocessing data in the 'data' folder.

Then you can train the model like this.
```
!python scripts/train.py    --train-path {"data/sentihood-train.tsv"}\
                            --dev-path {"data/sentihood-dev.tsv"} \
                            --bert-type {"base-bert"}\
                            --max-length 128 \
                            --output-dropout 0.1 \
                            --seed 0 \
                            --train-batch-size 32 \
                            --eval-batch-size 32 \
                            --max-epoch 20 \
                            --ffn-lr 0.001 \
                            --bert-lr 2e-5 

```
the detail of parameters are as follows:
```
"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --max-length=<int>                text length [default: 128]
    --output-dropout=<float>          prob of dropout applied to the output layer [default: 0.1]
    --seed=<int>                      fixed random seed number [default: 42]
    --train-batch-size=<int>          batch size [default: 32]
    --eval-batch-size=<int>           batch size [default: 32]
    --max-epoch=<int>                 max epoch [default: 20]
    --ffn-lr=<float>                  ffn learning rate [default: 0.001]
    --bert-lr=<float>                 bert learning rate [default: 2e-5]
    --bert-type=<str>                 language choice [default: base-bert]
    --dev-path=<str>                  file path of the dev set [default: '']
    --train-path=<str>                file path of the train set [default: '']
"""
```
After having already trained the model. We can validation it with our test dataset. Like this
 
```
!python scripts/test.py --test-path {'data/ABSA_15_Restaurants_Test.tsv'} \
                        --model-path {"2021-11-12-16:39:24_checkpoint.pt"} \
                        --max-length 160 \
                        --bert-type {"base-bert"}
```

the detailed of parameters are as follows:
```
"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --models-path=<str>                path of the trained models
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --bert-type=<str>                      language choice [default: base-bert]
    --test-path=<str>                 file path of the test set [default: ]
"""
```
