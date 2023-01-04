from scipy import stats
import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM


df = pd.DataFrame({'Position':np.repeat([1,2,3],9),
                    'Person':np.tile([1,2,3,4,5,6,7,8,9],3),
                    'power':[10,12,38,14,28,9,21,7,17,18,20,51,22,37,25,29,23,23,20,19,52,27,43,26,31,24,20]})
print(df)

print(AnovaRM(data=df, depvar='power',subject='Position', within=['Person']))