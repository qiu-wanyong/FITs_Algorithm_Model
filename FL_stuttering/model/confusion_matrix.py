# 混淆矩阵
import sys
sys.path.append(r'D:\Anaconda\Lib\site-packages')
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv('predicted_data3/50_3.csv', encoding= 'utf-8')

true = list(df['label'])
pred = list(df['predict_result'])

cm = confusion_matrix(true, pred)
ind = ['Pro', 'Nd', 'Int', 'Bl', 'Mod', 'Snd', 'Wd', 'UI']
cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm2 = cm2 * 100

df = pd.DataFrame(cm2, index=ind)
df.columns = ind




plt.rcParams['font.family'] = 'Times New Roman' # 设置字体样式
plt.rcParams['font.size'] = '24' # 设置字体大小
sns.heatmap(df, annot=True, cmap="Blues", fmt='.1f', annot_kws={"fontsize": 24})
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(rotation=45)

plt.show()