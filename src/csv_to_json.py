import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.converters import df2squad

df = pd.read_csv("data.csv", converters={"paragraphs": literal_eval}, encoding="utf-8")

# Converting dataframe to SQuAD format
json_data = df2squad(df=df, squad_version='v1.1', output_dir='.', filename="data")
