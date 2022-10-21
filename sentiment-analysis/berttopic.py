import pandas as pd
from cmath import isnan
from bertopic import BERTopic

df = pd.read_csv('')
df.head()

dataArray = []

for item in df['Review Summary']:
    if item != ' ':
        if type(item) == float:
            if isnan(item):
                continue
            else:
                dataArray.sppend(str(item))
                continue

        dataArray.append(item.strip())


topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")
