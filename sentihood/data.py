import csv
import json


def read(file_name):
    sentiments, sentences, ids, targets, aspects = [], [], [], [], []
    f = open(file_name)
    data = json.load(f)

    for dic in data:
        opinions = dic['opinions']
        temp_aspects, temp_targets, temp_sentiments = [], [], []

        for opinion in opinions:
            tar = opinion['target_entity']
            if tar not in temp_targets:
                temp_targets.append(tar)

        if len(temp_targets) == 1:
            for opinion in opinions:
                temp_aspects.append(opinion['aspect'])
                temp_sentiments.append(opinion['sentiment'].lower())
            if 'positive' in temp_sentiments and 'negative' in temp_sentiments:
                polarity = 'neutral'
            elif 'positive' in temp_sentiments:
                polarity = 'positive'
            else:
                polarity = 'negative'

            asp = ""
            for temp_aspect in temp_aspects[:-1]:
                asp += temp_aspect + ", "
            asp += 'and ' + temp_aspects[-1]
            aspects.append(asp)
            targets.append(temp_targets[0])
            sentiments.append(polarity)
            ids.append(dic['id'])
            sentences.append(dic['text'])
        else:
            continue
    # for sentiment in sentiments:
    #     if sentiment == 'neutral':
    #         print(sentiment)
    return ids, sentences, targets, aspects, sentiments


def write():
    for file in ['train', 'dev', 'test']:
        file_name = 'sentihood-' + file + '.json'
        ids, sentences, targets, aspects, sentiments = read(file_name)
        # ID	sentence	target	aspect	polarity
        with open('../data/sentihood-' + file + '.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['ID', 'sentence', 'target', 'aspect', 'polarity'])
            for _id, _sentence, _target, _aspect, _polarity in zip(ids, sentences, targets, aspects, sentiments):
                writer.writerow([_id, _sentence, _target, _aspect, _polarity])


write()
