import pandas as pd
f = pd.read_csv('D:/文件/大创/心音/feat_importance_aa.csv')['f']
t=pd.read_csv('D:/compare22-KSF/compare22-KSF/features/audeep/devel.csv')
d=pd.DataFrame()
d['filename']=t['filename']
d['label']=t['label']
for i in range(0,len(f)):
    d[f[i]]=t[f[i]]
print(d.info)
d.to_csv('D:/devel.csv',index=False)