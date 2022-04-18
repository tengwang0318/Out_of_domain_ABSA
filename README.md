# Out_of_domain_ABSA

[paper](https://arxiv.org/abs/2202.00484)

The repo code is based on [SpanEmo](https://github.com/hasanhuz/SpanEmo).

This repo will do "out of domain" aspect based sentiment analysis. It means that you will train a ABSA(Aspect bsaed sentiment analysis) model and use another dataset which is not relevant dataset, get the aspect and sentiment.

You can run the [notebook](https://colab.research.google.com/drive/1LfNqhqheVeY8YrsBI1l3IVlNQp_J6pl5?usp=sharing)

At first, you need to create a aspect detection model. I just use the [SpanEmo](https://github.com/hasanhuz/SpanEmo), this SpanEmo will get all of the aspects of sentence. Then you need to train the SpanEmo and build a model to output the aspect_polarity.

First of all, let's build the sentiment detection model. You need to load your data in "data" folder. You can see the data format with my preprocessing data in the 'data' folder.

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
Next, you need to train SpanEmo.

```
%cd SpanEmo
!python scripts/train.py    --train-path {"SemEval16/SemEval2016SpanEmoTrain.csv"}\
                            --dev-path {"SemEval16/SemEval2016SpanEmoValidation.csv"} \
                            --loss-type {'cross-entropy'} \
                            --max-length 128 \
                            --output-dropout 0.1 \
                            --seed 42 \
                            --train-batch-size 32 \
                            --eval-batch-size 32 \
                            --max-epoch 20 \
                            --ffn-lr 0.001 \
                            --bert-lr 2e-5 \
                            --lang {"English"} \
                            --alpha-loss 0.2 \
                            --bert-type {'BERT'} 
```

After training, you will get the model checkpoint, you need to load it when validation the model. 

```
!python scripts/test.py --test-path {'SemEval16/SevEval2016SpanEmo.csv'} \
                        --model-path {"/content/Out_of_domain_ABSA/SpanEmo/models/2021-11-28-08:09:30_checkpoint.pt"} \
                        --bert-type {"BERT"}
```

Then, you have got the aspects of sentence in predict.csv. You need to run this code, to get the right format of data that before feeding sentiment detection.
```
%cd scripts
!python data_integration.py
!python data_preprocess.py
!python create_all_aspect.py
%cd ../
```

Lastly, you can get the output of aspect and sentiment of sentences and validate the output.

```
%cd /content/Out_of_domain_ABSA
!python scripts/predict_test.py --models-path {"2021-11-28-07:28:04_checkpoint.pt"} \
                        --max-length 160 \
                        --bert-type {"base-bert"} \
                        --real-test-path {"data/semEval2016.tsv"} \
                        --fake-test-path {'data/fakerSemEval16.tsv'}
```

We have two baseline: 1. train and test sentimentPredictor with no aspect. 2. train and test sentimentPredictor with all aspect. 

1. train and test sentimentPredictor with no aspect
The input will be [CLS] + "what do you think of " + "NULL" + " of " + target  + [SEP] + sentence + [SEP].

2. train and test sentimentPredictor with all aspect
The input will be [CLS] + "what do you think of " + "all aspect"(like "food, restaurant, and drinks" ) + " of " + target + [SEP] + sentnece + [SEP].

Here is the result.


![image](https://user-images.githubusercontent.com/57594482/143766456-59bd0563-5bf8-4c98-9c5d-450f70921b68.png)

