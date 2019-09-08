import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('dataset/train.csv')

tb10 = pd.DataFrame(train_df.landmark_id.value_counts().tail(203094))
# tb10 = pd.DataFrame(train_df.landmark_id.value_counts().head(3094))
tb10.reset_index(level=0, inplace=True)
tb10.columns = ['landmark_id','count']
y = np.array(tb10['count'].tolist())
x = np.arange(203094)
plt.semilogy(x, y, c='r')
plt.ylabel('log count')
plt.show()
