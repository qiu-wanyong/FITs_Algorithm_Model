import matplotlib.pyplot as plt
UAR=[0.294,0.295,0.281,0.276,0.291]
f1=[0.304,0.308,0.290,0.287,0.310]
for i in [0,1,2,3,4]:
    UAR[i]=UAR[i]*100
    f1[i]=f1[i]*100
depth_trees = [2, 3, 4, 5, 6]
plt.rcParams['font.family'] = 'Times New Roman' # 设置字体样式
plt.rcParams['font.size'] = '46' # 设置字体大小
fig = plt.figure(figsize=(14,10))
plt.plot(depth_trees, UAR, 'b.-', depth_trees, f1, 'g.-')
plt.xlabel('Tree depth')
plt.ylabel('Metric values [%]')
plt.gcf().subplots_adjust(bottom=0.2,left=0.2)
plt.ylim(26, 32)
setx = [2, 3, 4, 5, 6]
plt.xticks(setx)
plt.legend([ 'UAR', 'UF1'])

for i in range(5):

    plt.text(depth_trees[i], UAR[i], "%.1f" %UAR[i], fontsize=38, verticalalignment="bottom",horizontalalignment="center")
    plt.text(depth_trees[i], f1[i], "%.1f" %f1[i], fontsize=38, verticalalignment="bottom",horizontalalignment="center")
plt.show()
