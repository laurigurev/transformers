import pandas as pd

data = pd.read_csv('dowloads/data/Reviews.csv')
data = data[:50]
text = data['Text']
text.to_csv('testdata.csv', index=False)