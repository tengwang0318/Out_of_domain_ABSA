import csv

with open('../predict.csv') as f:
    ids, texts, food, restaurant, atmosphere, drinks, location, service = [], [], [], [], [], [], [], []
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        ids.append(row[0])
        texts.append(row[1])
targets = []
with open('../../data/semEval2016.tsv') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        targets.append(row[2])
with open("../../data/with_all_aspect.tsv", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['ID', 'sentence', 'target', 'aspect', 'polarity'])
    for i in range(len(ids)):
        data = [ids[i], texts[i], targets[i], 'food, restaurant, atmosphere, drinks, location, and service', 'positive']
        writer.writerow(data)
with open("../../data/with_no_aspect.tsv", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['ID', 'sentence', 'target', 'aspect', 'polarity'])
    for i in range(len(ids)):
        data = [ids[i], texts[i], targets[i], 'Null', 'positive']
        writer.writerow(data)
