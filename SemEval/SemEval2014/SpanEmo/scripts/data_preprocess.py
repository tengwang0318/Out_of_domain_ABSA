import csv

with open('../predict.csv') as f:
    ids, texts, food, experience, service, atmosphere, price = [], [], [], [], [], [], []
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        ids.append(row[0])
        texts.append(row[1])
        food.append(row[2])
        experience.append(row[3])
        service.append(row[4])
        atmosphere.append(row[5])
        price.append(row[6])

targets = []
with open('../../semEval2014_test.tsv') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        targets.append(row[2])
# ID,text,food,experience,service,atmosphere,price
with open("../../fakerSemEval14.tsv", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['ID', 'sentence', 'target', 'aspect', 'polarity'])
    for i in range(len(ids)):

        temp_aspect = []
        if food[i] == 1:
            temp_aspect.append('food')
        if experience[i] == 1:
            temp_aspect.append('experience')
        if service[i] == 1:
            temp_aspect.append('service')
        if atmosphere[i] == 1:
            temp_aspect.append('atmosphere')
        if price[i] == 1:
            temp_aspect.append('price')
        aspect = ""
        if temp_aspect:
            for asp in temp_aspect[:-1]:
                aspect += asp + ", "
            aspect += "and " + temp_aspect[-1]
        else:
            aspect = "Null"
        writer.writerow([ids[i], texts[i], targets[i], aspect, 'positive'])
