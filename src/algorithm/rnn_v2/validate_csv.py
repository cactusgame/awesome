import csv


def validate_file(filename):
    with open(filename + ".new", 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        with open(filename) as text:
            # the `time` must be an integer
            # row = csv.reader(text, delimiter=',')
            # i = 0
            # for r in row:
            #     i = i + 1
            #     # if i > 100:
            #     #     break
            #     try:
            #         a = (int(r[0]))
            #     except:
            #         print(r[0])

            # validate the field count
            rows = csv.reader(text, delimiter=',')
            column_len = 0
            i = 0
            for r in rows:
                if i == 0:
                    column_len = len(r)
                i = i + 1
                try:
                    if i > 1:
                        time_is_int = (int(r[0]))
                except:
                    print "field time is not a int {}".format(r)

                try:
                    if len(r) == column_len:
                        writer.writerow(r)
                except:
                    print "error length {}".format(r)


# with open('/Users/happyelements/Documents/peng/awesome/feature_eval.csv') as text:
validate_file(filename='/Users/happyelements/Downloads/feature_eval.csv')
