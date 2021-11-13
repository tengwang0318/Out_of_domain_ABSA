import xml.etree.ElementTree as et
import csv


def load_data_Restaurant(file_name='Restaurants_Train_v2.xml'):
    parser = et.parse(file_name)
    root = parser.getroot()
    texts, aspects, polarities, ids = [], [], [], []
    for name in root.findall('sentence'):

        temp_aspect = []
        temp_text = name.find('text').text
        temp_polarity = []
        temp_id = name.get('id')
        for aspect in name.findall('aspectCategories'):
            for asp in aspect.findall('aspectCategory'):
                if asp.get("category") not in temp_aspect:
                    _aspect = asp.get('category')
                    if _aspect == 'anecdotes/miscellaneous':
                        _aspect = 'experience'
                    if _aspect == 'ambience':
                        _aspect = 'atmosphere'
                    temp_aspect.append(_aspect)
                    temp_polarity.append(asp.get('polarity'))
        texts.append(temp_text)
        aspects.append(temp_aspect)
        polarities.append(temp_polarity)
        ids.append(temp_id)

    return texts, aspects, polarities, ids


# texts, aspects, polarities, ids = load_data_Restaurant()
#
# with open("semEval2014.tsv", 'w') as f:
#     writer = csv.writer(f, delimiter='\t')
#     writer.writerow(['sentence_id', 'sentence', 'category', 'polarity', 'category_polarity', 'entailed'])
#     for i in range(len(texts)):
#         for j in range(len(aspects[i])):
#             for polar in ['negative', 'positive', 'neutral']:
#                 if polar == polarities[i][j]:
#                     writer.writerow([ids[i], texts[i], aspects[i][j], polar, aspects[i][j] + " " + polar, 'yes'])
#                 else:
#                     writer.writerow([ids[i], texts[i], aspects[i][j], polar, aspects[i][j] + " " + polar, 'no'])

texts, aspects, polarities, ids = load_data_Restaurant(file_name='restaurants-trial.xml')

with open('test_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'text', 'food', 'experience', 'service', 'atmosphere', 'price'])
    for i in range(len(texts)):
        data = [ids[i], texts[i], ]
        for asp in ['food', 'experience', 'service', 'atmosphere', 'price']:
            if asp in aspects[i]:
                data.append(1)
            else:
                data.append(0)
        writer.writerow(data)



with open("semEval2014.tsv", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['sentence_id', 'sentence', 'category', 'polarity', 'category_polarity', 'entailed'])
    for i in range(len(texts)):
        for j in range(len(aspects[i])):
            for polar in ['negative', 'positive', 'neutral']:
                if polar == polarities[i][j]:
                    writer.writerow([ids[i], texts[i], aspects[i][j], polar, aspects[i][j] + " " + polar, 'yes'])
                else:
                    writer.writerow([ids[i], texts[i], aspects[i][j], polar, aspects[i][j] + " " + polar, 'no'])

texts, aspects, polarities, ids = load_data_Restaurant(file_name='restaurants-trial.xml')

