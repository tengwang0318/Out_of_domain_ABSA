import xml.etree.ElementTree as et
import csv


def load_data(file_name):
    parser = et.parse(file_name)
    root = parser.getroot()
    ids, texts, aspects, sentiments, targets = [], [], [], [], []
    for review in root.findall('Review'):
        sentences = review.find('sentences')
        for sentence in sentences.findall('sentence'):
            opinions = sentence.find('Opinions')

            if opinions and len(opinions.findall('Opinion')) == 1:
                opinion = opinions.find("Opinion")

                targets.append(opinion.get('target'))
                ids.append(sentence.get('id'))
                texts.append(sentence.find('text').text)
                sentiments.append(opinion.get('polarity'))
                _asp = opinion.get('category').split('#')[0]
                aspects.append(_asp.lower())
                # if _asp == 'AMBIENCE':
                #     aspects.append('atmosphere')
                # else:
                #     aspects.append(_asp.lower())
    assert len(ids) == len(texts) == len(aspects) == len(sentiments) == len(targets)
    # return ids, targets, sentiments, texts, aspects
    return texts, aspects, ids


def write_data(file_name):
    texts, aspects, ids = load_data(file_name)
    fixed_aspect = ['service', 'hotel', 'rooms', 'facilities', 'location', 'rooms_amenities', 'food_drinks']
    with open('aspects_hotel.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['ID', 'text', 'service', 'hotel', 'rooms', 'facilities', 'location', 'rooms_amenities', 'food_drinks'])
        for _id, _text, _aspects in zip(ids, texts, aspects):
            temp = [_id, _text, ]
            for _asp in fixed_aspect:
                if _asp in _aspects:
                    temp.append(1)
                else:
                    temp.append(0)
            writer.writerow(temp)


load_data('ABSA15_Hotels_Test.xml')
write_data('ABSA15_Hotels_Test.xml')
