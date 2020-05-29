"""
Created on Wed Apr  8 11:38:40 2020
@author: DESHMUKH
LOGISTIC REGRESSION
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pylab
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split 
from sklearn import metrics
pd.set_option('display.max_columns',None)

# =========================================================================================
# Business Problem :- Prepare a prediction model for probability of extra marital affair.
# =========================================================================================

affairs = pd.read_csv("Affairs.csv",index_col = 0)
affairs.columns
affairs.info()
affairs.isnull().sum()
pd.value_counts(affairs['naffairs'].values)
affairs.shape

# Summary
affairs.describe()

# Converting numerical value in Binary of output
affairs.naffairs[affairs.naffairs > 0] = 1

# Percentage of Affairs and No Affairs
pd.value_counts(affairs['naffairs'].values)/len(affairs)*100 # 75% - No Affairs , 25 - Affairs

############################## - Exploratary Data Analysis - ##############################

# Measure of Central Tendancy / First moment business decision
affairs.mean() 
affairs.median()
affairs.mode()

# Mesaure of Dispersion / Secound moment business decision
affairs.var()
affairs.std()

# Skewness / Kurtosis - Third and Forth moment business decision
affairs.skew()
affairs.kurt()

# Graphical Representaion 
# Histogram
affairs.hist()

# Boxplot
plt.boxplot(affairs.naffairs)

# Heatmap plot
sns.heatmap(affairs.corr(),annot = True, annot_kws={"size": 5})

################################### - Splitting Data - ####################################

# Splitting Data in X and y
X = affairs.iloc[:,1:18]
y = affairs.iloc[:,0]

# Splitting Data in Train and Test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3 , random_state = False) # 30% Test data

########################## - Bulding Logistic Regression Model - ##########################

logit_model1 = smf.logit('y_train ~ vryunhap+unhap+avgmarr+hapavg+antirel+notrel+slghtrel+smerel', data = X_train).fit()
logit_model1.summary()
logit_model1.aic # 452

############################ - Accuracy of model by test data - ###########################

predict = logit_model1.predict(X_test)
predict

from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score

cnf_matrix = confusion_matrix(y_test, predict > 0.5 )
cnf_matrix

######################### - Accuracy , Sensitivity , Specificity - ########################

cnf_matrix
total1=sum(sum(cnf_matrix))
accuracy1=(cnf_matrix[0,0]+cnf_matrix[1,1])/total1
print ('Accuracy : ', accuracy1) # 79 %

sensitivity1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity : ', sensitivity1 ) # 99%

specificity1 = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity : ', specificity1)

###################################### -  ROC Curve - #####################################

fpr, tpr, threshold = metrics.roc_curve(y_test, predict)
roc_auc = metrics.auc(fpr, tpr)

#import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Auc = 0.75

                            #---------------------------------------------#









