# awesome

## Feature SDK
save all features in sqlite.    
`target` is also a `feature`

- save(date,share_id,feature_name,value) : value support String and float.
- update(date,share_id,feature_name,value)
- delete_row(date,share_id)
- get_row(date,share_id)
- get_rows_by_date(date): return all lines by day
- get_rows_by_share(share_id): return all lines by shares
- get_columns(columns)
- get_columns_by_conditions(columns,start_date=None,end_date=None,share_ids=None): return data which `date` >= start_time and <= `end_time` and share_id in `share_ids`


## Rec Log SDK
has implemnted in Feature SDK



## software version