import pandas as pd
import numpy as np

df = pd.read_csv('semEval2016.tsv', delimiter='\t')
sentiment = df.polarity.values
# print(sentiment[:10])
positive, negative, neutral = sentiment == 'positive', sentiment == 'negative', sentiment == 'neutral'
positive = positive.astype(float)
neutral = neutral.astype(float)
negative = negative.astype(float)
print(positive, negative, neutral)
print()
labels = np.stack((positive, neutral, negative), axis=1)
print(labels)

a = np.random.random((10, 3))
print(a)
print(a.max(axis=1))
print((a == a.max(axis=1, keepdims=1)).astype(float))
