import xml.etree.ElementTree as et
import csv
from collections import Counter



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

