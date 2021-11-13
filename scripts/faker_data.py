import csv

ids, sentences, aspects, polarity, aspect_polarity, entailed = [], [], [], [], [], []
with open("../SpanEmo/predict.csv") as f:
    reader = csv.reader(f)
    dic = {2: 'food', 3: 'experience', 4: 'service', 5: 'atmosphere', 6: 'price'}
    next(reader)
    for row in reader:
        for i in range(2, 7):
            if row[i] == '1.0':
                for polar in ['positive', 'negative', 'neutral']:
                    ids.append(row[0])
                    sentences.append(row[1])
                    aspect_polarity.append(dic[i] + " " + polar)
                    aspects.append(dic[i])
                    polarity.append(polar)

with open('../data/faker.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['sentence_id', 'sentence', 'category', 'polarity', 'category_polarity', 'entailed'])
    for i in range(len(ids)):
        writer.writerow([ids[i], sentences[i], aspects[i], polarity[i], aspect_polarity[i], 'no'])

