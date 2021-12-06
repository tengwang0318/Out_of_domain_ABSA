import xml.etree.ElementTree as et
import csv
from collections import Counter


def load_data_Restaurant(file_name):
    parser = et.parse(file_name)
    root = parser.getroot()
    ids, texts, aspects = [], [], []
    for review in root.findall('Review'):
        sentences = review.find('sentences')

        for sentence in sentences.findall('sentence'):
            temp_aspect = []
            opinions = sentence.find('Opinions')
            if opinions and len(opinions.findall('Opinion')) == 1:
                for opinion in opinions.findall('Opinion'):
                    temp = opinion.get('category').split("#")[0]
                    if temp not in temp_aspect:
                        if temp == 'AMBIENCE':
                            temp_aspect.append('atmosphere')
                        else:
                            temp_aspect.append(temp.lower())
                texts.append(sentence.find('text').text)
                aspects.append(temp_aspect)
                ids.append(sentence.get('id'))

    return texts, aspects, ids


def write_data(file_name):
    texts, aspects, ids = load_data_Restaurant(file_name)
    fixed_aspect = ['restaurant', 'service', 'food', 'atmosphere', 'drinks', 'location']
    with open('aspects_restaurant.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'text', 'food', 'restaurant', 'service', 'atmosphere', 'drinks', 'location'])
        for _id, _text, _aspects in zip(ids, texts, aspects):
            temp = [_id, _text, ]
            for _asp in fixed_aspect:
                if _asp in _aspects:
                    temp.append(1)
                else:
                    temp.append(0)
            writer.writerow(temp)


write_data('ABSA-15_Restaurants_Train_Final.xml')
