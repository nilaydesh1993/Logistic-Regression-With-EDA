
"""
Created on Fri Apr 10 19:52:05 2020
@author: DESHMUKH
LOGISTIC REGRESSION 
"""
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pylab as plt
import pylab

from sklearn.model_selection import train_test_split
from sklearn import metrics
pd.set_option('display.max_columns',None)

# =======================================================================================================
# Business Problem :- Prepare a prediction model Whether the client has subscribed a term deposit or not 
# =======================================================================================================

bankdata = pd.read_csv("bank_data.csv")
bankdata.isnull().sum()
bankdata.columns
bankdata = bankdata.rename({'joadmin.':'joadmin','joblue.collar':'joblue_collar','joself.employed':'joself_employed'}, axis=1)
bankdata.shape
bankdata.columns

# head
bankdata.head(10)

# Describe
bankdata.describe()

############################## - Exploratary Data Analysis - ##############################

#sns.pairplot(bankdata,hue = 'y')

bankdata.y.value_counts()
sns.countplot(bankdata.y , data = bankdata)

# Percentile of subscribed or not-subscribed
countsub = len(bankdata[bankdata.y == 1])
countnonsub = len(bankdata[bankdata.y == 0])

subscribed = countsub/(countnonsub + countsub)       #subscirbe (1) = 11.7%
non_subscibe = countnonsub/(countnonsub + countsub)  #subscirbe (0) = 88.3%

# Indepedent variable exploration
# Histogram
plt.hist(bankdata.age)
plt.hist(bankdata.duration)
plt.hist(bankdata.balance)
plt.hist(bankdata.campaign)
plt.hist(bankdata.pdays)

# Boxplot
plt.boxplot(bankdata.age)
plt.boxplot(bankdata.duration)
plt.boxplot(bankdata.balance)
plt.boxplot(bankdata.campaign)
plt.boxplot(bankdata.pdays)

# Heatmap
sns.heatmap(bankdata.corr(),annot = True, annot_kws={"size": 4})

################################### - Splitting Data - ####################################

# Splitting in X and y
X = bankdata.iloc[:,0:31] 
y = bankdata['y']

# Spliting in Train and Test
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3)

########################## - Bulding Logistic Regression Model - ##########################

model1 = smf.logit('y_train ~ age + default + balance + housing + loan + duration + campaign + pdays + previous + poutother + poutsuccess + poutunknown + con_telephone + con_unknown + divorced+ married + joadmin + joblue_collar + joentrepreneur + johousemaid + jomanagement + joretired + joself_employed + joservices + jostudent + jotechnician + jounemployed',data = X_train).fit()
model1.summary() # Bulding after removing dummy variable trap
model1.aic # 15870

# In order to take care of multi collinearity,we remove variables having high vif
# For each X, calculate VIF and save in dataframe
from statsmodels.stats.outliers_influence import variance_inflation_factor
#vif = pd.DataFrame()
#vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X_train.shape[1])]
#vif["features"] = X_train.columns
#vif.round(1) 
""" this vif step is not work here because data set is have dummy variable trap problem """

# Building model by deleting most insignificant variable at time till all model significant 
model2 = smf.logit('y_train ~ balance + housing + loan + duration + campaign + poutother + poutsuccess + poutunknown + con_telephone + con_unknown + divorced+ married + joadmin + joblue_collar + joentrepreneur + johousemaid + jomanagement + joretired + joself_employed + joservices + jostudent + jotechnician + jounemployed',data = X_train).fit()
model2.summary() # Bulding after removing dummy variable trap
model2.aic # 15867 , after deleting pdays,age,defualt, previous
# by comparing two model we can observe that second model is better amoung both


############################ - Accuracy of model by test data - ###########################

predict = model2.predict(X_test)
predict

from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score
cnf_matrix = confusion_matrix(y_test, predict > 0.5)
cnf_matrix

################# - Accuracy,Sensitivity,Specificity by confusion matrix - ################

cnf_matrix
total1=sum(sum(cnf_matrix))
accuracy1=(cnf_matrix[0,0]+cnf_matrix[1,1])/total1
print ('Accuracy : ', accuracy1)    # 0.90 %

sensitivity1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity : ', sensitivity1 ) # 0.90 %

specificity1 = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity : ', specificity1)  # 0.31 %

###################################### -  ROC Curve - #####################################

fpr, tpr, threshold = metrics.roc_curve(y_test, predict)
roc_auc = metrics.auc(fpr, tpr)

# import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Area under curve  = 0.89

                        #---------------------------------------------#

