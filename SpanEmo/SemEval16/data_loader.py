import csv
import xml.etree.ElementTree as et


def read(file_name):
    parser = et.parse(file_name)
    root = parser.getroot()
    aspects, targets, ids, polarities, texts = [], [], [], [], []
    for review in root.findall('Review'):
        sentences = review.find('sentences')
        for sentence in sentences.findall('sentence'):
            temp_id = sentence.get('id')
            temp_text = sentence.find('text').text
            opinions = sentence.find('Opinions')

            if opinions:
                temp_targets = set()
                for opinion in opinions.findall("Opinion"):
                    temp_target = opinion.get('target')
                    if temp_target not in temp_targets:
                        temp_targets.add(temp_target)
                if len(temp_targets) > 1:
                    continue

                temp_aspects, temp_polarities = [], []
                for opinion in opinions.findall("Opinion"):
                    temp_aspect = opinion.get('category').split('#')[0].lower()
                    if temp_aspect == 'ambience':
                        temp_aspect = 'atmosphere'
                    temp_polarity = opinion.get('polarity')

                    temp_aspects.append(temp_aspect)
                    temp_polarities.append(temp_polarity)
                if 'positive' in temp_polarities and 'negative' in temp_polarities:
                    temp_polarity = 'neutral'
                elif 'positive' in temp_polarities and 'negative' not in temp_polarities:
                    temp_polarity = 'positive'
                elif 'positive' not in temp_polarities and 'negative' in temp_polarities:
                    temp_polarity = 'negative'
                else:
                    temp_polarity = 'neutral'
                aspects.append(temp_aspects)
                targets.append(temp_target)
                polarities.append(temp_polarity)
                texts.append(temp_text)
                ids.append(temp_id)
    # for i in range(10):
    #     print(ids[i], texts[i], aspects[i], targets[i], polarities[i])

    return ids, texts, aspects, targets, polarities


def write_data_SpanEmo(file_name, output_file):
    ids, texts, aspects, targets, polarities = read(file_name)
    fixed_aspects = {'food', 'restaurant', 'atmosphere', 'drinks', 'location', 'service'}
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'text', 'food', 'restaurant', 'atmosphere', 'drinks', 'location', 'service'])
        for _ids, _text, _aspects in zip(ids, texts, aspects):
            data = [_ids, _text]
            for asp in ['food', 'restaurant', 'atmosphere', 'drinks', 'location', 'service']:
                if asp in _aspects:
                    data.append(1)
                else:
                    data.append(0)
            writer.writerow(data)


write_data_SpanEmo("ABSA16_Restaurants_Train_SB1_v2.xml", 'SevEval2016SpanEmo.csv')
write_data_SpanEmo('restaurants_trial_english_sl.xml', 'test.csv')


def write_data_SP(file_name):
    ids, texts, aspects, targets, polarities = read(file_name)
    with open("../../data/semEval2016.tsv", 'w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerow(['ID', 'sentence', 'target', 'aspect', 'polarity'])
        for _id, _text, _aspect, _target, _polarity in zip(ids, texts, aspects, targets, polarities):
            aspect = ""
            if len(_aspect) > 1:
                for asp in _aspect[:-1]:
                    aspect += asp + ", "
                aspect += 'and ' + _aspect[-1]
            else:
                aspect = _aspect[0]
            writer.writerow([_id, _text, _target, aspect, _polarity])
write_data_SP('ABSA16_Restaurants_Train_SB1_v2.xml')