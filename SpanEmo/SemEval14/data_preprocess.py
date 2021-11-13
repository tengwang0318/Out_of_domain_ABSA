import xml.etree.ElementTree as et
import csv
from collections import Counter


# for laptop dataset
def load_data_laptop(file_name):
    parser = et.parse(file_name)
    root = parser.getroot()
    texts, aspects = [], []
    for name in root.findall('sentence'):
        # print(name.find('text').text)
        temp_aspect = []
        temp_text = name.find('text').text
        # if name.find('aspectTerms'):
        for aspect in name.findall('aspectTerms'):
            for asp in aspect.findall('aspectTerm'):
                temp_aspect.append(asp.get('term'))
        texts.append(temp_text)
        aspects.append(temp_aspect)

    return texts, aspects


texts, aspects = load_data_laptop('Laptop_Train_v2.xml')
assert len(texts) == len(aspects)

# check how many aspects does the dataset have.
counter = Counter()
for aspect in aspects:
    for item in aspect:
        counter[item] += 1

print(counter)
print(len(texts))


def load_data_Restaurant(file_name='Restaurants_Train_v2.xml'):
    parser = et.parse(file_name)
    root = parser.getroot()
    texts, aspects = [], []
    for name in root.findall('sentence'):
        # print(name.find('text').text)
        temp_aspect = []
        temp_text = name.find('text').text
        # if name.find('aspectCategories'):
        for aspect in name.findall('aspectCategories'):
            for asp in aspect.findall('aspectCategory'):
                temp_aspect.append(asp.get('category'))
        texts.append(temp_text)
        aspects.append(temp_aspect)

    return texts, aspects


texts, aspects = load_data_Restaurant()
print(aspects)
counter = Counter()
for aspect in aspects:
    for item in aspect:
        counter[item] += 1

print(counter)
print(len(texts))


def write_data(file_name='Restaurants_Train_v2.xml', output_file='aspects_restaurants.csv'):
    fixed_aspect = ['food', 'anecdotes/miscellaneous', 'service', 'ambience', 'price']
    texts, aspects = load_data_Restaurant(file_name)
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'text', 'food', 'experience', 'service', 'atmosphere', 'price'])
        ID = 0
        for text, asp in zip(texts, aspects):
            data = [ID, text]
            for item in fixed_aspect:
                if item in asp:
                    data.append(1)
                else:
                    data.append(0)
            writer.writerow(data)
            ID += 1


write_data('Restaurants_Train_v2.xml')
write_data("restaurants-trial.xml", 'test_data.csv')
"""
import pandas as pd

df = pd.read_csv('../SemEval14/aspect_restaurants_train.csv')
x_train, y_train = df.iloc[:, 1].values, df.iloc[:, 2:].values
print(x_train, y_train)
segment_a = "food experience service atmosphere or price?"
label_names = ['food', 'experience', 'service', 'atmosphere', 'price']
# segment_a = "anger anticipation disgust fear joy love optimism hopeless sadness surprise or trust?"
# label_names = ["anger", "anticipation", "disgust", "fear", "joy",
#                "love", "optimism", "hopeless", "sadness", "surprise", "trust"]
"""
from transformers import AutoTokenizer

segment_a = "1 2 3 or 4"
label_names = ["1", '2', '3', '4']
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
sentences = ["I am sad! I can't fix this bug!"]
for x in sentences:
    x = tokenizer.encode_plus(segment_a, x)
    input_id = x['input_ids']
    print(input_id)

    print(tokenizer.convert_ids_to_tokens(input_id))
    # label indices
    label_idxs = [tokenizer.convert_ids_to_tokens(input_id).index(label_names[idx])
                  for idx, _ in enumerate(label_names)]
    print(label_idxs)
