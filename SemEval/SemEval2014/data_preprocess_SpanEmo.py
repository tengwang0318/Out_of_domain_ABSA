import xml.etree.ElementTree as et
import csv
from collections import Counter


def load_data_Restaurant(file_name='Restaurants_Train 2.xml'):
    parser = et.parse(file_name)
    root = parser.getroot()
    texts, aspects, ids = [], [], []
    for name in root.findall('sentence'):
        # print(name.find('text').text)
        aspectTerms = name.find('aspectTerms')

        if aspectTerms and len(aspectTerms.findall('aspectTerm')) == 1:
            temp_aspect = []
            temp_text = name.find('text').text
            # if name.find('aspectCategories'):
            for aspect in name.findall('aspectCategories'):
                for asp in aspect.findall('aspectCategory'):
                    temp_aspect.append(asp.get('category'))
            texts.append(temp_text)
            ids.append(name.get('id'))
            aspects.append(temp_aspect)

    return texts, aspects, ids


texts, aspects, ids = load_data_Restaurant()
print(aspects)
counter = Counter()
for aspect in aspects:
    for item in aspect:
        counter[item] += 1

print(counter)
print(len(texts))


def write_data(file_name='Restaurants_Train_v2.xml', output_file='aspects_restaurants.csv'):
    fixed_aspect = ['food', 'anecdotes/miscellaneous', 'service', 'ambience', 'price']
    texts, aspects, ids = load_data_Restaurant(file_name)
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'text', 'food', 'experience', 'service', 'atmosphere', 'price'])
        for _id, text, asp in zip(ids, texts, aspects):
            data = [_id, text]
            for item in fixed_aspect:
                if item in asp:
                    data.append(1)
                else:
                    data.append(0)
            writer.writerow(data)


write_data('Restaurants_Train 2.xml')
