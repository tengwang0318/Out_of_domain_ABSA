import xml.etree.ElementTree as et
import csv


def load_data(file_name):
    parser = et.parse(file_name)
    root = parser.getroot()
    ids, targets, aspects, sentences, sentiments = [], [], [], [], []
    for sentence in root.findall('sentence'):

        aspectTerms = sentence.find('aspectTerms')

        if aspectTerms and len(aspectTerms.findall('aspectTerm')) == 1:
            aspectCategories = sentence.find('aspectCategories')
            temp_aspects = []
            for aspectCategory in aspectCategories.findall('aspectCategory'):
                if aspectCategory.get('category') == 'anecdotes/miscellaneous':
                    temp_aspects.append('experience')
                elif aspectCategory.get('category') == 'ambience':
                    temp_aspects.append('atmosphere')
                else:
                    temp_aspects.append(aspectCategory.get('category'))

            ids.append(sentence.get('id'))
            targets.append(aspectTerms.find('aspectTerm').get('term'))
            sentiments.append(aspectTerms.find('aspectTerm').get('polarity'))
            sentences.append(sentence.find('text').text)
            aspects.append(temp_aspects)

    return ids, targets, sentiments, sentences, aspects


# ID	sentence	target	aspect	polarity
def write_data(file_name):
    ids, targets, sentiments, sentences, aspects = load_data(file_name)
    with open('semEval2014.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['ID', 'sentence', 'target', 'aspect', 'polarity'])
        for _id, _target, _sentiment, _sentence, _aspects in zip(ids, targets, sentiments, sentences, aspects):
            if len(_aspects) == 1:
                aspect = _aspects[0]
            else:
                aspect = ", ".join([_aspects[i] for i in range(len(_aspects) - 1)])
                aspect += ', and ' + _aspects[-1]
            writer.writerow([_id, _sentence, _target, aspect, _sentiment])


write_data('Restaurants_Train 2.xml')
