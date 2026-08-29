from src.hopsworks_utils import get_or_create_feature_group

fg = get_or_create_feature_group()
df = fg.read()
print(f"Total rows: {len(df)}")
print(f"Latest datetime_utc: {df['datetime_utc'].max()}")
print(df.sort_values('datetime_utc').tail(5)[['event_id', 'datetime_utc']])