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
    print(set(aspects))
    return ids, targets, sentiments, texts, aspects
    # return texts, aspects, ids


# ID	sentence	target	aspect	polarity
def write_data(file_name):
    ids, targets, sentiments, sentences, aspects = load_data(file_name)
    with open('hotel.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['ID', 'sentence', 'target', 'aspect', 'polarity'])
        for _id, _target, _sentiment, _sentence, _aspects in zip(ids, targets, sentiments, sentences, aspects):
            writer.writerow([_id, _sentence, _target, _aspects, _sentiment])


write_data('ABSA15_Hotels_Test.xml')
