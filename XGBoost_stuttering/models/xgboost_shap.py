import sys
sys.path.append(r'D:\Anaconda\Lib\site-packages')
import lightgbm as lgb
import xgboost as xgb
import shap
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

train = pd.read_csv('../data/train(train_80%).csv')  # take row 1-256 as training set
test = pd.read_csv('../data/test(train_20%).csv')  # take row 257-366 as testing set

#df1.to_csv('D:/train_final.csv')
#df2.to_csv('D:/devel_final.csv')
train_X = train.iloc[:,2:]
train_Y = train.iloc[:,1]

test_X = test.iloc[:,2:]
test_Y = test.iloc[:,1]



# 加载model
xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 1
param['max_depth'] = 5
#param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 8
watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 50
bst = xgb.train(param, xg_train, num_round, watchlist);
# get prediction
pred = bst.predict(xg_test)


# model导入SHAP
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(train_X)
n = np.array(shap_values)
print(n.shape)   # (8, 994, 61) (类别数，样本数，特征数)
shap.summary_plot(shap_values, train_X, plot_type="bar", max_display=10,class_names=['Pro','Nd','Int','BI','Mod','Snd','Wd','UI']) #针对所有类别的解释
shap.summary_plot(shap_values[0], train_X, plot_type="bar", max_display=10,class_names=['Pro']) #针对类别为0的那一类的解释
'''


'''
# 用model对测试集进行预测
y_pred_prob = bst.predict(xg_test)
y_pred = np.argmax(y_pred_prob, axis=1) #找出最大概率的下标，即为哪一类
y_pred = list(y_pred)
y_true = list(test_Y)

UAR = recall_score(y_true, y_pred, average='macro')
UF1 = f1_score(y_true, y_pred, average='macro')

print(UAR,UF1)
