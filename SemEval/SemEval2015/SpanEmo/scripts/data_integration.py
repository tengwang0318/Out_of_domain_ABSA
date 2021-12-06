import csv
from collections import defaultdict

id_to_text = defaultdict(list)
with open("../../aspect_test.csv.tsv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        id_to_text[row[0]].append(row[1])

with open('../predict.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:

        id_to_text[row[0]].append(row[2])
        id_to_text[row[0]].append(row[3])
        id_to_text[row[0]].append(row[4])
        id_to_text[row[0]].append(row[5])
        id_to_text[row[0]].append(row[6])
        id_to_text[row[0]].append(row[7])
# print(id_to_text)
# food,restaurant,service,atmosphere,drinks,location
with open('../predict.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['ID', 'text', 'food', 'restaurant', 'service', 'atmosphere', 'drinks', 'location'])
    for _id in id_to_text.keys():
        writer.writerow([_id] + id_to_text[_id])

