import csv
from collections import defaultdict

all_aspects = ['safety', 'general', 'price', 'live', 'transit-location', 'quiet', 'shopping', 'dining',
               'nightlife', 'multicultural', 'green-nature', 'touristy']


def write_all_aspect_and_no_aspect():
    for file in ['train', 'dev', 'test']:
        ids, sentences, targets, aspects, polarities = load_data("sentihood-" + file + '.tsv')
        with open('all_aspects_sentihood-' + file + ".tsv", 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['ID', 'sentence', 'target', 'aspect', 'polarity'])
            fixed_aspects = ""
            for asp in all_aspects[:-1]:
                fixed_aspects += asp + ", "
            fixed_aspects += 'and ' + all_aspects[-1]
            for i in range(len(ids)):
                writer.writerow([ids[i], sentences[i], targets[i], fixed_aspects, polarities[i]])
        with open('no_aspect_sentihood-' + file + '.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['ID', 'sentence', 'target', 'aspect', 'polarity'])
            for i in range(len(ids)):
                writer.writerow([ids[i], sentences[i], targets[i], "Null", polarities[i]])

def load_data(file_name):
    ids, sentences, targets, aspects, polarities = [], [], [], [], []
    with open(file_name) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            ids.append(row[0])
            sentences.append(row[1])
            targets.append(row[2])
            aspects.append(row[3])
            polarities.append(row[4])
    return ids, sentences, targets, aspects, polarities


write_all_aspect_and_no_aspect()