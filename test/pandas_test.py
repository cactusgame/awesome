import pandas as pd

key_date1 = ["2018-1-1", "2018-1-2", "2018-1-3"]
key_share_id1 = ["a1", "a2", "a3"]
feature_1 = ["x1", "x2", "x3"]

key_date2 = ["2018-1-1", "2018-1-2"]
key_share_id2 = ["a1", "a2"]
feature_2 = ["y1", "y2"]

df1 = pd.DataFrame({"key": key_date1, "feature1": feature_1, "share_id": key_share_id1})
print (df1)

df2 = pd.DataFrame({"key": key_date2, "feature2": feature_2, "share_id": key_share_id2})
print (df2)

print(pd.merge(df1, df2, on=["key","share_id"]))
