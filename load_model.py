import sys
sys.path.append(r'D:\Anaconda\Lib\site-packages')
import lightgbm as lgb
import shap
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

df1 = pd.read_csv('train_guest.csv', encoding='utf-8')
df2 = pd.read_csv('test.csv', encoding='utf-8')

y1, y2 = df1['y'], df2['y']
X1, X2 = df1.iloc[:, 2:], df2.iloc[:, 2:]



# 加载model
gbm = lgb.Booster(model_file = 'model in lgb.txt')
gbm.params["objective"] = "multiclass"


'''
# model导入SHAP
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(X1)
n = np.array(shap_values)
print(n.shape)   # (8, 994, 61) (类别数，样本数，特征数)
shap.summary_plot(shap_values, X1, plot_type="bar", max_display=10) #针对所有类别的解释
shap.summary_plot(shap_values[0], X1, plot_type="bar", max_display=10) #针对类别为0的那一类的解释
'''


'''
# 用model对测试集进行预测
y_pred_prob = gbm.predict(X2)
y_pred = np.argmax(y_pred_prob, axis=1) #找出最大概率的下标，即为哪一类
y_pred = list(y_pred)
y_true = list(y2)

UAR = recall_score(y_true, y_pred, average='macro')
UF1 = f1_score(y_true, y_pred, average='macro')

print(UAR,UF1)'''