import pandas as pd

df = pd.read_csv('sentihood-train.tsv', sep='\t')
print(df.sentence.values)
print(df.category_polarity.values)
print(df.entailed.values)
a = df.entailed.values == 'yes'
print(a.astype(int))
import torch

a = torch.randn((32, 128))
print(a)
print(a[:, 0].shape)
