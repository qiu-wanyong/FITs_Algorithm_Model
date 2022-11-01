import matplotlib.pyplot as plt
UAR=[0.281,0.282,0.275,0.284,0.295,0.295,0.290]
f1=[0.292,0.294,0.285,0.296,0.308,0.308,0.303]
for i in [0,1,2,3,4,5,6]:
    UAR[i]=UAR[i]*100
    f1[i]=f1[i]*100
depth_trees = [10,20,30,40,50,60,70]
plt.rcParams['font.family'] = 'Times New Roman' # 设置字体样式
plt.rcParams['font.size'] = '46' # 设置字体大小
fig = plt.figure(figsize=(14,10))
plt.plot( depth_trees, UAR, 'b.-', depth_trees, f1, 'g.-')
plt.xlabel('Tree number')
plt.ylabel('Metric values [%]')
plt.ylim(26, 32)  # 设置y轴的数值显示范围
setx = [10,20,30,40,50,60,70]
plt.xticks(setx)
plt.legend([ 'UAR', 'UF1'])
plt.gcf().subplots_adjust(bottom=0.2,left=0.2)
for i in range(7):

    plt.text(depth_trees[i], UAR[i], "%.1f" %UAR[i], fontsize=38, verticalalignment="bottom",horizontalalignment="center")
    plt.text(depth_trees[i], f1[i], "%.1f" %f1[i], fontsize=38, verticalalignment="bottom",horizontalalignment="center")
plt.show()
