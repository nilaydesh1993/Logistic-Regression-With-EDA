"""
Created on Sun Apr 12 11:58:28 2020
@author: DESHMUKH
LOGISTIC REGRESSION 
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pylab
import statsmodels.formula.api as smf
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn import metrics

# ==========================================================================================================
# Business Problem :- Prepare a prediction model Whether political candidate wins an election (0-win,1-loss)
# ==========================================================================================================

election = pd.read_csv("election_data.csv")
election.shape
election.isnull().sum()
election = election.drop(election.index[0]) # Removing empty row
#election = election.iloc[1:11,:]
election.columns
election = election.iloc[:,1:5] # Removing ID column
election.columns = "result","year","amountspend","popularityrank" 
election.columns

election.head(5)
election.describe()

############################## - Exploratary Data Analysis - ##############################

# Scatter plot
sns.pairplot(election,hue = 'result')

# Histogram
election.hist()

# Boxplot
plt.boxplot(election.year)
plt.boxplot(election.amountspend)
plt.boxplot(election.popularityrank)

# Determining Properties of Y

# Count of Loss and Win
sns.countplot(election.result, data = election)
election.result.value_counts()

# Percentise of win and loss
countwin = len(election[election.result == 0])
countloss = len(election[election.result == 1])
win = countwin/(countwin+countloss)     # 40 %
loss = countloss/(countwin+countloss)   # 60 %

# Heatmap
sns.heatmap(election.corr(),annot = True)

################################### - Splitting Data - ####################################

# Splitting into X and Y
X = election.iloc[:,1:4]
y = election["result"]

################################### - Model Bulding - #####################################

# We can use either one step form below two to train our model

# 1 - By Statemodel
model1 = smf.logit('y ~ amountspend + popularityrank', data = X).fit()
model1.summary()
model1.aic
predict = model1.predict(X)
predict
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score
cnf_matrix = confusion_matrix(y, predict > 0.5)
cnf_matrix


# 2 - By Sklearn
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X,y)
predictions = logmodel.predict(X)
print(confusion_matrix(y,predictions))

################# - Accuracy,Sensitivity,Specificity by confusion matrix - #################

cnf_matrix
total1=sum(sum(cnf_matrix))
accuracy1=(cnf_matrix[0,0]+cnf_matrix[1,1])/total1
print ('Accuracy : ', accuracy1)    # 0.90 %

sensitivity1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity : ', sensitivity1 ) # 0.75 %

specificity1 = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity : ', specificity1)  # 0.100 %

###################################### -  ROC Curve - #####################################

fpr, tpr, threshold = metrics.roc_curve(y, predict)
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

# Area under curve  = 0.96
                
                    #--------------------------------------------#
