import csv

city_path = ["calgary-receiving-2019.csv", "hamilton-recieving-2019.csv"]

yards = [0,0]
yards_idx = 0

for p in city_path:
	with open(p) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
				line_count += 1
				yards[yards_idx] += int(row[7])
	yards_idx += 1

print(yards[0])
print(yards[1])