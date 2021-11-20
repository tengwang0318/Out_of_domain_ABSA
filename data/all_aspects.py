import csv
from collections import defaultdict

all_aspects = ['safety', 'general', 'price', 'live', 'transit-location', 'quiet', 'shopping', 'dining',
               'nightlife', 'multicultural', 'green-nature', 'touristy']


def load_data(file_name):
    data = defaultdict(lambda: defaultdict(list))
    with open(file_name) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            data[row[0]]['sentence'].append(row[1])
            data[row[0]]['category'].append(row[2])
            data[row[0]]['polarity'].append(row[3])
            data[row[0]]['entailed'].append(row[5])
    return data


def write():
    for file in ['train', 'dev', 'test']:
        data = load_data('sentihood-' + file + '.tsv')
        with open('all_aspects_sentihood_' + file + '.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['sentence_id', 'sentence', 'category', 'polarity', 'category_polarity', 'entailed'])
            for _id, values in data.items():
                for i in range(len(values['sentence'])):
                    writer.writerow([_id, values['sentence'][i], values['category'][i], values['polarity'][i],
                                     values['category'][i] + " " + values['polarity'][i], values['entailed'][i]])
                for aspect in all_aspects:
                    if aspect not in values['category']:
                        for polar in ['positive', 'negative', 'neutral']:
                            writer.writerow([_id,
                                             values['sentence'][0],
                                             aspect,
                                             polar,
                                             aspect + " " + polar,
                                             'no'])


write()
