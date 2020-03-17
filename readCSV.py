import csv

array = []
with open('testing.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # print(row)
        array.append(row)
print((array))