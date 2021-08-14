#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # 檢視數據

# In[2]:


train_data = pd.read_csv(r"C:\Users\adsad\OneDrive\Desktop\kaggle\house-price\train.csv")
train_data


# In[3]:


test_data = pd.read_csv(r"C:\Users\adsad\OneDrive\Desktop\kaggle\house-price\test.csv")
test_data


# In[4]:


train_data.info()


# In[5]:


test_data.info()


# In[6]:


train_dtype = train_data.dtypes
train_dtype.value_counts()


# In[7]:


test_dtype = test_data.dtypes
test_dtype.value_counts()


# In[8]:


train = train_data.drop("SalePrice",axis = 1)
all_data= pd.concat((train, test_data)).reset_index(drop=True)
all_data.index = all_data["Id"]
all_data = all_data.drop("Id",axis =1)
all_data


# # 補缺值

# In[9]:


import matplotlib.pyplot as plt


# In[10]:


import missingno as msno


# In[11]:


missing = all_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True,ascending = False)
plt.subplots(figsize=(15,5)) # 設定畫面大小
missing.plot.bar()
plt.show()


# In[12]:


msno.matrix(all_data.iloc[:,:40])
plt.show()


# In[13]:


msno.matrix(all_data.iloc[:,40:])
plt.show()


# In[14]:


all_data.isnull().sum().sort_values(ascending = False).head(35)


# In[15]:


null_rate = all_data.isnull().sum() / len(all_data) *100
null_rate.sort_values(ascending = False).head(35)


# In[16]:


#從缺值約50%之上的特徵值先去判斷

#PoolQC:游泳池等級,MiscFeature:其他類別未涵蓋的其他功能,Alley:胡同類型,Fence:圍欄等級,FireplaceQu:壁爐等級
for col in ("PoolQC","MiscFeature","Alley","Fence","FireplaceQu"):
    all_data[col] = all_data[col].fillna("None")


# In[17]:


#LotFrontage:與房子相連的街道距離，可能與同區域間的房子相似，所以拿Neighborhood同區的中間值當作替代值
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# In[18]:


#GarageCond:車庫條件,GarageQual:車庫等級,GarageFinish:車庫內部裝潢,GarageType:車庫位置
for col in ("GarageCond","GarageQual","GarageFinish","GarageType"):
     all_data[col] = all_data[col].fillna("None")
    
#GarageYrBlt:車庫建照年份,GarageArea:車庫面積,GarageCars:車庫可容納車子的數量(為數值，所以缺值直接補0)
for col in ("GarageYrBlt","GarageArea","GarageCars"):
    all_data[col] = all_data[col].fillna(0)


# In[19]:


#BsmtCond : 地下室的狀況,BsmtExposure:地下室採光程度,BsmtQual:地下室高度,BsmtFinType1:地下室1的等級,BsmtFinType2:地下室2的等級
for col in ("BsmtCond","BsmtExposure","BsmtQual","BsmtFinType1","BsmtFinType2"):
     all_data[col] = all_data[col].fillna("None")
#BsmtFinSF1:地下室1的面積,BsmtFinSF2:地下室2的面積,BsmtHalfBath:地下室半浴室數量,BsmtFullBath:地下室全浴室數量,BsmtUnfSF:未完成的地下室面積,TotalBsmtSF:地下室總面積(為數值，所以缺值直接補0)
for col in ("BsmtFinSF1","BsmtFinSF2","BsmtHalfBath","BsmtFullBath","BsmtUnfSF","TotalBsmtSF",):
    all_data[col] = all_data[col].fillna(0)


# In[20]:


#MasVnrType:砌體材質
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
#MasVnrArea:砌體面積
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


# In[21]:


#MSZoning:銷售的住宅類型，正常應該要有值，所以這裡應該是真的遺漏掉的值，並非沒有的狀況，所以拿最多的RL(低密度住宅區)來代替
all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data["MSZoning"].mode()[0])


# In[22]:


#Utilities:水電等公共設施種類，雖然公共設施越多應該要月貴，但該特徵值除了1個NoSeWa以及2個nan外，其餘都是AllPub，而NoSeWa正好又在訓練集，對於我們預測的數據沒有幫助，所以可以直接刪除這個特徵值
all_data =  all_data.drop(["Utilities"],axis=1) 


# In[23]:


#Functional:家電功能等級，Typ為典型的，也就是一般的，所以2個缺值的部分可以補成Typ
all_data["Functional"] = all_data["Functional"].fillna("Typ")


# In[24]:


#Exterior1st,Exterior2nd:房屋外牆飾面，只有1個缺值，所以用最多的數值補
all_data["Exterior1st"] = all_data["Exterior1st"].fillna(all_data["Exterior1st"].mode()[0])
all_data["Exterior2nd"]= all_data["Exterior2nd"].fillna(all_data["Exterior2nd"].mode()[0])


# In[25]:


#SaleType:銷售類型，缺值一樣只有一個，所以用最普遍的WD(常規的契約)來替補
all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].mode()[0])


# In[26]:


#Electrical:電力系統，缺值只有一個，所有用最普遍的SBrkr(標準的電力系統)來替補
all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].mode()[0])


# In[27]:


#KitchenQual:廚房等級，缺值只有一個，所以用最普遍的TA(典型(中等)的等級)來替補
all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data["KitchenQual"].mode()[0])


# In[28]:


all_data.isnull().sum().sort_values(ascending = False)


# # 數值與非數值分開判斷

# In[29]:


object_feature = all_data[all_data.dtypes[all_data.dtypes == object].index]
train_object = object_feature[:len(train_data)]
train_object 


# In[30]:


train_y = pd.DataFrame(train_data["SalePrice"])
train_y.index = train_data["Id"]
train_object["SalePrice"] = train_y
train_object


# In[31]:


import seaborn as sns


# In[32]:


for col in train_object .columns[:-1]:
        plt.subplots(figsize=(20,5))
        sns.boxplot(x=col, y="SalePrice", data=train_object )
        plt.title('Boxplot for {}'.format(col))
        plt.show()


# In[33]:


for col in train_object.columns:
    data = train_object.groupby([col])[["SalePrice"]].agg(['mean','median','count'])
    sort_data = data.sort_values(data.columns[1],ascending = True)
    print(sort_data)


# In[34]:


MSZoning_dict = {"C (all)":1,
                "RM":2,"RH":2,
                "RL":3,
                "FV":4}
#+-------------------------------------------------------
Street_dict = {"Grvl":1,
               "Pave":2}
#+-------------------------------------------------------
Alley_dict = {"Grvl":1,
        "None":2,"Pave":2}
#+-------------------------------------------------------
LotShape_dict = {"Reg":1,
               "IR1":2,"IR3":2,
                "IR2":3}
#+-------------------------------------------------------
LandContour_dict = {"Bnk":1,
                    "Lvl":2,
                   "Low":3,
                    "HLS":4}
#+-------------------------------------------------------
LotConfig_dict = {"Inside":1,"Corner":1,"FR2":1,
                 "FR3":2,"CulDSac":2}                               
#+-------------------------------------------------------
LandSlope_dict = {"Gtl":1,
                  "Sev":2,"Mod":2}
#+-------------------------------------------------------
Neighborhood_dict = {"MeadowV":1,"IDOTRR":1,"BrDale":1,"OldTown":1,"Edwards":1,"BrkSide":1, 
                    "Sawyer":2,"Blueste":2,"SWISU":2,"NAmes":2,"NPkVill":2,"Mitchel":2,
                     "SawyerW":3,"Gilbert":3,"NWAmes":3,"Blmngtn":3,"CollgCr":3,
                     "ClearCr":4,"Crawfor":4,"Veenker":4,"Somerst":4,"Timber":4,
                    "StoneBr":5,"NoRidge":5,"NridgHt":5}
#-------------------------------------------------------
Condition1_dict = {"Artery":1,"Feedr":1,"RRAe":1,
                  "Norm":2,"RRAn":2,"RRNe":2,
                  "RRNe":3,"PosN":3,"PosA":3,"RRNn":3}
#-------------------------------------------------------
Condition2_dict = {"RRNn":1,"Artery":1,"Feedr":1,"RRAn":1,
                   "Norm":2,
                   "RRAe":3,"PosN":3,"PosA":3}
#-------------------------------------------------------
BldgType_dict = {"2fmCon":1,
                "Duplex":2,"Twnhs":2,
                "1Fam":3,"TwnhsE":3}
#+-------------------------------------------------------
HouseStyle_dict = {"1.5Unf":1,
                   "1.5Fin":2,"2.5Unf":2,"SFoyer":2,
                  "1Story":3,"SLvl":3,
                  "2Story":4,"2.5Fin":4}
#-------------------------------------------------------
RoofStyle_dict = {"Gambrel":1,
                 "Gable":2,"Mansard":2,
                 "Hip":3,"Flat":3,"Shed":3}
#-------------------------------------------------------
RoofMatl_dict = {"Roll":1,"ClyTile":1,"CompShg":1,
                "Tar&Grv":2,"Metal":2,
                 "Membran":3,"WdShake":3,"WdShngl":3}
#-------------------------------------------------------
Exterior1st_dict = {"BrkComm":1,"AsphShn":1,"CBlock":1,"AsbShng":1,
                   "WdShing":2,"Wd Sdng":2,"MetalSd":2,
                   "Stucco":3,"HdBoard":3,
                   "BrkFace":4,"Plywood":4,
                   "VinylSd":5,"CemntBd":5,"Stone":5,"ImStucc":5}
#-------------------------------------------------------
Exterior2nd_dict = {"CBlock":1,"AsbShng":1,"Wd Sdng":1,"Wd Shng":1,"MetalSd":1,"AsphShn":1,
                   "Stucco":2,"Brk Cmn":2,"HdBoard":2,"BrkFace":2,"Plywood":2,"Stone":2,
                   "ImStucc":3,"VinylSd":3,"CmentBd":3,"Other":3}
#-------------------------------------------------------
MasVnrType_dict = {"BrkCmn":1,"None":1,
                  "BrkFace":2,
                  "Stone":3}
#+-------------------------------------------------------
ExterQual_dict = {"Fa":1,
                  "TA":2,
                  "Gd":3,
                  "Ex":4}      
#+-------------------------------------------------------
ExterCond_dict = {"Po":1,
                 "Fa":2,
                 "Gd":3,
                 "Ex":4,"TA":4}                             
#+-------------------------------------------------------
Foundation_dict = {"Slab":1,
                  "BrkTil":2,"Stone":2,
                  "CBlock":3,
                  "Wood":4,
                  "PConc":5}
#-------------------------------------------------------
BsmtQual_dict = {"None":1,
                "Fa":2,
                "TA":3,
                "Gd":4,
                "Ex":5}                    
#+-------------------------------------------------------
BsmtCond_dict = {"Po":1,
                "None":2,
                "Fa":3,
                "TA":4,
                "Gd":5}                             
#+-------------------------------------------------------
BsmtExposure_dict = {"None":1,
                     "No":2,
                     "Mn":3,"Av":3,
                     "Gd":4}                           
#+-------------------------------------------------------
BsmtFinType1_dict = {"None":1,
                   "LwQ":2,"BLQ":2,"Rec":2,
                   "ALQ":3,"Unf":3,
                   "GLQ":4}                             
#+-------------------------------------------------------
BsmtFinType2_dict = {"None":1,
                    "BLQ":2,"Rec":2,"LwQ":2,
                    "Unf":3,
                    "ALQ":4,
                    "GLQ":5}
#-------------------------------------------------------
Heating_dict = {"Floor":1,"Grav":1,
               "Wall":2,"OthW":2,
               "GasW":3,
               "GasA":4}
#-------------------------------------------------------
HeatingQC_dict = {"Po":1,
                 "Fa":2,
                 "TA":3,
                 "Gd":4,
                 "Ex":5}                          
#+-------------------------------------------------------
CentralAir_dict = {"N":1,
                  "Y":2}         
#+-------------------------------------------------------
Electrical_dict = {"Mix":1,
                   "FuseP":2,
                  "FuseF":3,
                  "FuseA":4,
                  "SBrkr":5}                           
#+-------------------------------------------------------
KitchenQual_dict = {"Fa":1,
                   "TA":2,
                   "Gd":3,
                   "Ex":4}                             
#+-------------------------------------------------------
Functional_dict = {"Maj2":1,"Sev":1,
                  "Mod":2,"Min1":2,"Min2":2,"Maj1":2,
                  "Typ":3}
#-------------------------------------------------------
FireplaceQu_dict = {"Po":1,"None":1,
                   "Fa":2,
                   "TA":3,
                   "Gd":4,
                   "Ex":5}
#-------------------------------------------------------
GarageType_dict = {"None":1,"CarPort":1,
                  "Detchd":2,
                  "Basment":3,"2Types":3,
                  "Attchd":4,
                  "BuiltIn":5}
#-------------------------------------------------------
GarageFinish_dict = {"None":1,
                    "Unf":2,
                    "RFn":3,
                    "Fin":4}                             
#+-------------------------------------------------------
GarageQual_dict = {"Po":1,"None":1,
                  "Fa":2,
                  "Ex":3,
                  "TA":4,
                  "Gd":5}
#-------------------------------------------------------
GarageCond_dict = {"None":1,"Po":1,
                  "Fa":2,"Ex":2,
                  "Gd":3,
                  "TA":4}
#-------------------------------------------------------
PavedDrive_dict = {"N":1,
                  "P":2,
                  "Y":3}              
#+-------------------------------------------------------
PoolQC_dict = {"None":1,
              "Gd":2,
              "Fa":3,
              "Ex":4}                             
#+-------------------------------------------------------
Fence_dict = {"MnWw":1,
             "MnPrv":2,"GdWo":2,
             "GdPrv":3,
             "None":4}                             
#+-------------------------------------------------------
MiscFeature_dict = {"Othr":1,
                    "Shed":2,
                    "None":3,"Gar2":3,
                    "TenC":4}
#+-------------------------------------------------------
SaleType_dict = {"Oth":1,
                "ConLI":2,"COD":2,"ConLD":2,"ConLw":2,
                "WD":3,
                "CWD":4,"New":4,"Con":4}
#-------------------------------------------------------
SaleCondition_dict = {"AdjLand":1,
                     "Abnorml":2,"Family":2,
                     "Alloca":3,"Normal":3,
                     "Partial":4}


# In[35]:


new_object_feature =object_feature.replace({"MSZoning":MSZoning_dict,"Street":Street_dict,"Alley":Alley_dict,"LotShape":LotShape_dict,
                                         "LandContour":LandContour_dict,"LotConfig":LotConfig_dict,"LandSlope":LandSlope_dict,"Neighborhood": Neighborhood_dict,
                                         "Condition1":Condition1_dict,"Condition2":Condition2_dict,"BldgType":BldgType_dict,"HouseStyle":HouseStyle_dict,
                                         "RoofStyle":RoofStyle_dict,"RoofMatl":RoofMatl_dict,"Exterior1st":Exterior1st_dict,"Exterior2nd":Exterior2nd_dict,
                                         "MasVnrType":MasVnrType_dict,"ExterQual":ExterQual_dict,"ExterCond":ExterCond_dict,"Foundation":Foundation_dict,
                                         "BsmtQual":BsmtQual_dict,"BsmtCond":BsmtCond_dict,"BsmtExposure":BsmtExposure_dict,"BsmtFinType1":BsmtFinType1_dict,
                                         "BsmtFinType2":BsmtFinType2_dict,"Heating":Heating_dict,"HeatingQC":HeatingQC_dict,"CentralAir":CentralAir_dict,
                                         "Electrical":Electrical_dict,"KitchenQual":KitchenQual_dict,"Functional":Functional_dict,"FireplaceQu":FireplaceQu_dict,
                                         "GarageType":GarageType_dict,"GarageFinish":GarageFinish_dict,"GarageQual":GarageQual_dict,
                                         "GarageCond":GarageCond_dict,"PavedDrive":PavedDrive_dict,"PoolQC":PoolQC_dict,"Fence":Fence_dict,
                                         "MiscFeature":MiscFeature_dict,"SaleType":SaleType_dict,"SaleCondition":SaleCondition_dict})
new_object_feature


# In[36]:


classification_int_float = ["MSSubClass","OverallQual","OverallCond"]


# In[37]:


int_float_feature = all_data[all_data.dtypes[all_data.dtypes != object].index]
all_int_float = int_float_feature.drop(classification_int_float,axis = 1)
train_int_float = all_int_float[:len(train_data)]
train_int_float["SalePrice"] = train_y
train_int_float


# In[38]:


for col in train_int_float.columns[:-1]:
    sns.scatterplot(x=col, y='SalePrice', data=train_int_float)
    plt.title('Scatterplot for {}'.format(col))
    plt.show()


# In[39]:


import numpy as np


# In[40]:


log_all_int_float = np.log1p(all_int_float)
log_all_int_float


# In[41]:


classificantion_feature = int_float_feature[classification_int_float]
classificantion_feature_train = classificantion_feature[:len(train)]
classificantion_feature_train["SalePrice"] = train_y
classificantion_feature_train


# In[42]:


for col in classificantion_feature_train.columns:
    data = classificantion_feature_train.groupby([col])[["SalePrice"]].agg(['mean','median','count'])
    sort_data = data.sort_values(data.columns[1],ascending = True)
    print(sort_data)


# In[43]:


MSSubClass_dict = {180:1,30:1,45:1,190:1,
                  50:2,90:2,85:2,40:2,160:2,
                  70:3,20:3,
                   75:4,80:4,
                  120:5,60:5,
                  150:0}
#-------------------------------------------------------
OverallCond_dict = {1:1,
                   2:2,3:2,
                   4:3,
                   8:4,6:4,7:4,
                   9:5,5:5}


# In[44]:


classificantion_feature["MSSubClass"] =classificantion_feature["MSSubClass"].map(MSSubClass_dict)
classificantion_feature["OverallCond"] = classificantion_feature["OverallCond"].map(OverallCond_dict)


# In[45]:


int_float_feature


# In[46]:


all_feature = pd.concat( [new_object_feature, log_all_int_float,classificantion_feature], axis=1 )
all_feature


# # 數據分割

# In[47]:


#將數據分割回原本的train data跟test data
train_x = all_feature[:len(train_data)]
test_x = all_feature[len(train_data):]


# In[48]:


train_x.shape,test_x.shape


# # 預測目標標準化

# In[49]:


from scipy.stats import norm
from scipy import stats


# In[50]:


sns.distplot(train_y['SalePrice'] , fit=norm);
#Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_y['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_y['SalePrice'], plot=plt)
plt.show()


# In[51]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train_y["SalePrice"] = np.log1p(train_y["SalePrice"])
#Check the new distribution 
sns.distplot(train_y['SalePrice'] , fit=norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_y['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_y['SalePrice'], plot=plt)
plt.show()


# In[52]:


test_y = pd.read_csv(r"C:\Users\adsad\OneDrive\Desktop\kaggle\house-price\sample_submission.csv",index_col = "Id")
test_y


# In[53]:


test_y["SalePrice"] = np.log1p(test_y["SalePrice"])
sns.distplot(test_y['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(test_y['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
res = stats.probplot(test_y['SalePrice'], plot=plt)
plt.show()


# # 特徵篩選(隨機森林)

# In[54]:


from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor


# In[55]:


from sklearn.model_selection import cross_val_score


# In[56]:


def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


# In[57]:


train_y_v = train_y.values.ravel()
test_y_v= test_y.values.ravel()


# In[58]:


train_x.shape,train_y_v.shape,test_x.shape,test_y_v.shape


# In[59]:


#隨機迴歸森林
rfr = RandomForestRegressor(n_jobs=-1, n_estimators=100)
rfr.fit(train_x,train_y_v)

#極端迴歸森林
etr = ExtraTreesRegressor()
etr.fit(train_x,train_y_v)

#梯度提升迴歸樹
gbr = GradientBoostingRegressor()
gbr.fit(train_x,train_y_v)


# In[60]:


name = ["RandomForestRegressor","ExtraTreesRegressor","GradientBoostingRegressor"]
model = [rfr,etr,gbr]
for n,m in zip(name,model):
    predict_y = m.predict(test_x)
    predict = pd.DataFrame(predict_y,index = test_y.index,columns = ["SalePrice"])
    score = rmse_cv(m,train_x ,train_y.values.ravel())
    print("{} mean score : {:.4f} ({:.4f})".format(n,score.mean(),score.std()))


# In[61]:


plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

feat_importances = pd.Series(gbr.feature_importances_, index= train_x.columns)
feat_importances.nlargest(10).plot(kind='barh')


# In[62]:


final_feature = train_x [feat_importances.nlargest(10).index]
final_feature


# # 模型訓練

# In[63]:


from sklearn.linear_model import Lasso,ElasticNet,BayesianRidge
import xgboost as xgb
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import  GradientBoostingRegressor
import lightgbm as lgb
from  sklearn.svm import SVR


# In[64]:


lasso = Lasso()
ENet = ElasticNet()
XGB_Model = xgb.XGBRegressor()
KRR = KernelRidge()
GBoost = GradientBoostingRegressor()
lgb_Model = lgb.LGBMRegressor()
SVR = SVR()
Bay = BayesianRidge()


# In[65]:


import warnings
warnings.filterwarnings("ignore")


# In[66]:


name = ["Lasso","ElasticNet","XGBoost","KernelRidge","GradientBoostingRegressor","Lightgbm","SVR",
        "RandomForestRegressor","ExtraTreesRegressor","GradientBoostingRegressor"]
model = [lasso,ENet,XGB_Model,KRR,GBoost ,lgb_Model,SVR,Bay,rfr,gbr,etr]
for n,m in zip(name,model):
    m.fit(train_x,train_y)
    train_score = m.score(train_x,train_y)
    score = rmse_cv(m,train_x ,train_y.values.ravel())
    print("{} mean score(Standard Deviation score) : {:.4f} ({:.4f})".format(n,score.mean(),score.std()))


# In[70]:


predict = GBoost.predict(test_x)
predict = pd.DataFrame(predict,index = test_y.index,columns = ["SalePrice"])
predict_true = np.expm1(predict)
predict_true


# In[71]:


predict_true.to_csv(r"C:\Users\adsad\OneDrive\Desktop\kaggle\house-price\predict.csv")


# In[ ]:




