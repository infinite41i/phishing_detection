from scipy.io import arff
import pandas as pd

data = arff.loadarff("./Training_Dataset.arff")
df = pd.DataFrame(data[0])
print(df.head())