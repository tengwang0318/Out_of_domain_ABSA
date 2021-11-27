import numpy as np
from torch.utils.data import Dataset
from transformers import BertModel, AutoTokenizer, AutoModelWithLMHead, AutoModel
from tqdm import tqdm
import torch
import pandas as pd


class DataClass(Dataset):
    def __init__(self, args, file_name):
        self.args = args
        self.file_name = file_name
        self.max_length = int(args['--max-length'])
        self.sentences, self.aspects, self.labels, self.targets = self.load_dataset()

        if args['--bert-type'] == 'base-bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif args['--bert-type'] == 'DeBerta':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        elif args['--bert-type'] == 'RoBerta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.inputs = self.process_data()
        self.labels = torch.tensor(self.labels)

    def load_dataset(self):
        df = pd.read_csv(self.file_name, sep='\t')
        sentences, aspects, sentiment, targets = df.sentence.values, df.aspect.values, df.polarity.values, df.target.values

        positive, negative, neutral = sentiment == 'positive', sentiment == 'negative', sentiment == 'neutral'
        positive = positive.astype(float)
        neutral = neutral.astype(float)
        negative = negative.astype(float)
        labels = np.stack((positive, neutral, negative), axis=1)

        return sentences, aspects, labels, targets

    def process_data(self):
        desc = "Preprocessing dataset {}...".format("")
        inputs = []
        # for tar in self.targets:
        #     print(tar)

        # import time
        # time.sleep(10)
        for x, aspect, target in tqdm(zip(self.sentences, self.aspects, self.targets), desc=desc):
            if not pd.isna(target):
                x = self.tokenizer.encode_plus(
                    "what do you think of " + aspect + " of " + target,
                    x,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    pad_to_max_length=True,
                    truncation=True
                )
            else:
                x = self.tokenizer.encode_plus(
                    "what do you think of " + aspect + " of NULL" ,
                    x,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    pad_to_max_length=True,
                    truncation=True
                )
            input_id = x['input_ids']
            inputs.append(input_id)
        inputs = torch.tensor(inputs, dtype=torch.long)
        return inputs

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        return inputs, labels

    def __len__(self):
        return len(self.inputs)
