
import pandas as pd

class LabelEncoder:
    def __init__(self, x_range=201, y_range=201):
        self.encoder = {
            (x, y): y + x * x_range
            for x in range(x_range)
            for y in range(y_range)
        }
        self.decoder = {value: key for key, value in self.encoder.items()}
    def encode(self, x):
        return self.encoder[x]
    def decode(self, x):
        return self.decoder[x]
    def transform(self, row):
        return self.encode((row['x'], row['y']))
    
def data_impute(df, col='uid'):
    part_df = []
    for idx, group in df.groupby(col):
        part_df.append(group.ffill().bfill())
    return pd.concat(part_df)

