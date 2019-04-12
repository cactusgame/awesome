import csv

# target_key = 'ror_20_days'
# new_target_key = 'ror_20_days_bool'
#
# # output
# output = open('data/test.csv', 'w')
# fieldnames = ['share_id', 'ror_20_days', 'close_b1', 'close_b0', 'ror_60_days', 'close_b2', 'ror_10_days', 'ror_40_days', 'ror_05_days', 'time', 'ror_20_days_bool']
# writer = csv.DictWriter(output, fieldnames=fieldnames)
# writer.writeheader()
#
# f = open('data/features.csv', 'rb')
# reader = csv.DictReader(f)
#
# for row in reader:
#     ror_20_days_bool = None
#     if row[target_key] > 0:
#         ror_20_days_bool = 1
#     else:
#         ror_20_days_bool = 0
#     row[new_target_key] = ror_20_days_bool
#
#     # writer.writeheader()
#     writer.writerow(row)
# f.close()
# output.close()

# target_key = 'ror_20_days'
# new_target_key = 'ror_20_days_bool'
#
# # output
# output = open('data/test.csv', 'w')
# writer = csv.writer(output, delimiter=',')
#
# f = open('data/features.csv', 'rb')
# reader = csv.reader(f, delimiter=',')
#
# header = None
# target_index = -1
# for row in reader:
#     if header is None:
#         header = row
#         target_index = header.index(target_key)
#         row.append(new_target_key)
#     else:
#         r = 1 if float(row[target_index]) > 0 else 0
#         row.append(r)
#     writer.writerow(row)
# f.close()
# output.close()

with open('/Users/happyelements/Documents/peng/awesome/data/features.csv.eval','r') as f:
    sum = 0
    i = 0
    for line in f:
        print line[-3:-2]
        sum = sum + int(line[-3:-2])
        i = i + 1

    print "mean = {}".format( sum/(i*1.0))
    print "len = {}".format(i)
