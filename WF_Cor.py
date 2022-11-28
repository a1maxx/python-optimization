import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

con = pd.read_csv("C:\\Users\\Administrator\\Desktop\\WF\\Training_R-197119_Candidate Attach #1_JDSE_SRF #462.csv")
con['rep_education'] = con['rep_education'].astype('category')
print(con.columns)



con = con.drop(['rep_education'],axis=1)
con.describe(include='category')
cormat = con.corr()
dfCorr = con.corr()
round(cormat,2)
sns.heatmap(cormat)
plt.show()

filteredDf = dfCorr[((dfCorr >= .7) | (dfCorr <= -.7)) & (dfCorr != 1.000)]
filteredDf2 = filteredDf.dropna()

plt.figure(figsize=(10,5))
sns.heatmap(filteredDf, annot=True, fmt='.2g',cmap= 'coolwarm')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()
plt.savefig('C:\\Users\\Administrator\\Desktop\\WF\\corHeat.png', dpi =800,bbox_inches='tight')

import seaborn as sns
sns.heatmap(con.corr(), annot = True, fmt='.2g',cmap= 'coolwarm')


components = list()
visited = set()
print(con.columns)
for col in con.columns:
    if col in visited:
        continue

    component = set([col, ])
    just_visited = [col, ]
    visited.add(col)
    while just_visited:
        c = just_visited.pop(0)
        for idx, val in dfCorr[c].items():
            if abs(val) > 0.999 and idx not in visited:
                just_visited.append(idx)
                visited.add(idx)
                component.add(idx)
    components.append(component)

for component in components:
    plt.figure(figsize=(12,8))
    sns.heatmap(dfCorr.loc[component, component], cmap="Reds")
    plt.show()