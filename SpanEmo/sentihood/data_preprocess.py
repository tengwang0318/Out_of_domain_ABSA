import json
import csv
import re
from collections import Counter


def load_data(file_name='sentihood-test.json'):
    f = open(file_name)
    data_json = json.load(f)

    sentences, aspects = [], []
    for item in data_json:
        opinions = item['opinions']
        temp_aspects = []
        if opinions:
            for opinion in opinions:
                temp_aspects.append(opinion['aspect'])
            sentences.append(item['text'].strip())
            aspects.append(temp_aspects)
    return sentences, aspects



_, aspects = load_data()
counter = Counter()
for item in aspects:
    for aspect in item:
        counter[aspect] += 1
print(counter.keys())
def write_data():
    fixed_aspects = ['safety', 'general', 'price', 'live', 'transit-location', 'quiet', 'shopping', 'dining',
                     'nightlife', 'multicultural', 'green-nature', 'touristy']
    for tmp_file in ['train', 'dev', 'test']:
        sentences, aspects = load_data('sentihood-' + tmp_file + '.json')
        with open('sentihood-' + tmp_file + '.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['ID', 'texts'] + fixed_aspects)
            ID = 0
            for sentence in sentences:
                data = [ID, sentence]
                tmp_aspects = aspects[ID]
                for aspect in fixed_aspects:
                    if aspect in tmp_aspects:
                        data.append(1)
                    else:
                        data.append(0)

                writer.writerow(data)
                ID += 1


write_data()
