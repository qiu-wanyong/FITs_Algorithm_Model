'''import sys
sys.path.append(r'D:\Anaconda\Lib\site-packages')'''
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from datetime import datetime
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

# 利用xgb.train中的get_score得到weight，gain，以及cover
params = {'max_depth': 3,  # 构建树的深度，越大越容易过拟合
          'n_estimators': 40,  # 树的个数
          'learning_rate': 0.3,  # 如同学习率
          'nthread': 4,
          'num_class': 8,
          'subsample': 1.0,  # 随机采样训练样本 训练实例的子采样比
          'colsample_bytree': 1,  # 生成树时进行的列采样
          'min_child_weight': 3,  # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
          # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
          # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
          # 'eval_metric' : ['logloss'],
          'objective':'multi:softmax',
          'seed': 1301}  # 随机种子

train = pd.read_csv('D:/compare22-KSF/compare22-KSF/features/audeep/train.csv')
val = pd.read_csv('D:/compare22-KSF/compare22-KSF/features/audeep/train.csv')

train_y, val_y = train['label'], val['label']
train_X, val_X = train.iloc[:, 2:], val.iloc[:, 2:]

xgtrain = xgb.DMatrix(train_X, label=train_y)
xgval = xgb.DMatrix(val_X, label=val_y)

# bst = xgb.train(params, xgtrain, num_boost_round=50)
model = xgb.train(params,
                  dtrain=xgtrain,
                  verbose_eval=True,
                  evals=[(xgtrain, "train"), (xgtrain, "valid")],
                  early_stopping_rounds=10,  # 当loss在10轮迭代之内，都没有提升的话，就stop。
                  num_boost_round=50
                  )

# ***************自定义特征重要性指标***************
# 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
# 在所有树中，某特征被⽤来分裂节点的次数
# 表⽰在所有树中，某特征在每次分裂节点时处理(覆盖)的所有样例的数量
# cover = total_cover / weight
# 在所有树中，某特征在每次分裂节点时带来的总增益
# gain = total_gain / weight
# 在平时的使⽤中，多⽤total_gain来对特征重要性进⾏排序
importance_eval_list = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
for i, importance_type in enumerate(importance_eval_list):
    feat_importance = model.get_score(importance_type=importance_type)
    feat_importance = pd.DataFrame.from_dict(feat_importance, orient='index')
    feat_importance.columns = [importance_type]
    if i == 0:
        df_temp = feat_importance
    else:
        df_temp = pd.merge(df_temp, feat_importance, how='outer', left_index=True, right_index=True)
        print('%s: ' % importance_type, model.get_score(importance_type=importance_type))

print('特征重要性结果为:\n', df_temp)

# 用shap包进行可视化
#print(model)

'''
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(train_X)
shap.summary_plot(shap_values, train_X, max_display=2)
'''

# 生成csv

model_suffix0 = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
feat_importance_name = 'feat_importance_b_%s.csv' % (model_suffix0)


feat_importance_name = 'feat_importance_aa.csv'
df_temp.to_csv(feat_importance_name, index=True)
print('*' * 10, '完成评估特征重要性!  详见文件{}'.format(feat_importance_name), '*' * 10)


# 用传统python包的matplotlib可视化
'''
zhibiao = df_temp.loc[:, 'total_gain']
print(zhibiao)
paixu = zhibiao.sort_values(ascending=True)
'''
# print(paixu)


'''
plt.figure()
paixu.plot(kind='barh', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.tick_params(labelsize=5)
plt.xlabel('total_gain')
plt.ylabel('feature names')
setx = np.arange(0, 5, 0.1)
plt.xticks(setx)
plt.grid(True, axis='both')
plt.show()
'''
