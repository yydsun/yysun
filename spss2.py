import pandas as pd #导入pandas包
from scipy import stats
df = pd.read_excel(r'/homes/yysun/Data/SPSS/Table25-6.1.xlsx')

leveneTestRes1 = stats.levene(df.loc[df.loc[:,'angle']=='90','pron'],df.loc[df.loc[:,'angle']=='45','pron'],center='mean') #进行方差齐性检验
statistic,pvalue=leveneTestRes1
print(df)
print(df.loc[df.loc[0:7],'pron'])
'''
print(statistic, pvalue)
leveneTestRes2 = stats.levene(df.loc[df.loc[:,'角度']=='90','neut'],df.loc[df.loc[:,'角度']=='45','neut'],center='mean')
print(leveneTestRes2)
leveneTestRes3 = stats.levene(df.loc[df.loc[:,'角度']=='90','supin'],df.loc[df.loc[:,'角度']=='45','supin'],center='mean')
print(leveneTestRes3)

df2 = df.melt(id_vars=['ID','角度'])
#print(df2)

import pingouin
aov = pingouin.mixed_anova(data= df2,dv='value',within='variable',subject='ID',between='角度')
print(aov.round(3))

from statsmodels.formula.api import ols 
from statsmodels.stats.anova import anova_lm 
formula = 'power~ angle + position ' 
anova_results = anova_lm(ols(formula,df).fit()) 
print(anova_results)
'''