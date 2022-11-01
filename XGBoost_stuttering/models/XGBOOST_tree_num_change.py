from sklearn.model_selection import train_test_split
import pandas as pd
train = pd.read_csv('../data/train(train_80%).csv')
test = pd.read_csv('../data/test(train_20%).csv')

#df1.to_csv('D:/train_final.csv')
#df2.to_csv('D:/devel_final.csv')
train_X = train.iloc[:,2:]
train_Y = train.iloc[:,1]

test_X = test.iloc[:,2:]
test_Y = test.iloc[:,1]

import xgboost as xgb
xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
accuracy=[]
UAR=[]
f1=[]
for i in [10,20,30,40,50,60]:
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
    num_round = i
    bst = xgb.train(param, xg_train, num_round, watchlist);
    # get prediction
    pred = bst.predict(xg_test)

    import numpy as np
    error_rate = np.sum(pred != test_Y) / len(test_Y)
    accuracy_rate=1-error_rate
    accuracy.append(accuracy_rate)
    print('Test accuracy using softmax = {}'.format(accuracy_rate))

    predl=pred.tolist()
    test_Yl=test_Y.values.tolist()
    from sklearn.metrics import recall_score, f1_score
   # print(recall_score(test_Yl,predl,average='macro'))
    UAR.append(recall_score(test_Yl,predl,average='macro'))
    f1.append(f1_score(test_Yl,predl,average='macro'))
import matplotlib
import matplotlib.pyplot as plt
for i in [0,1,2,3,4,5]:
    UAR[i]=UAR[i]*100
    f1[i]=f1[i]*100
depth_trees = [10,20,30,40,50,60]
plt.rcParams['font.family'] = 'Times New Roman' # 设置字体样式
plt.rcParams['font.size'] = '46' # 设置字体大小
fig = plt.figure(figsize=(14,10))
plt.plot( depth_trees, UAR, 'b.-', depth_trees, f1, 'g.-')
plt.xlabel('Tree number')
plt.ylabel('Metric values [%]')
#plt.ylim(25, 30)  # 设置y轴的数值显示范围
setx = [10,20,30,40,50,60]
plt.xticks(setx)
plt.legend([ 'UAR', 'UF1'])
plt.gcf().subplots_adjust(bottom=0.2,left=0.2)
for i in range(len(accuracy)):

    plt.text(depth_trees[i], UAR[i], "%.1f" %UAR[i], fontsize=38, verticalalignment="bottom",horizontalalignment="center")
    plt.text(depth_trees[i], f1[i], "%.1f" %f1[i], fontsize=38, verticalalignment="bottom",horizontalalignment="center")
plt.show()
