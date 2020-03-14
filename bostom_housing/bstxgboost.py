import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.preprocessing import LabelEncoder,RobustScaler,StandardScaler
from scipy.stats import skew
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import cross_val_score,GridSearchCV,KFold
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR,LinearSVR
from xgboost import XGBRegressor




#查看各个条件于售价的关系
def datasee(datas):
    for i in range(1,datas.shape[1]-2,8):
        sns.pairplot(x_vars=datas.columns[i:i+8],
                     y_vars=datas.columns[-1],data=datas,
                     dropna=True
                     )
        plt.show()
#查看目标分布
def saleprice_see(datas):
    plt.hist(x=datas['SalePrice'],bins=datas.shape[0],color='steelblue',)
    plt.show()

def data_value_deal(datas):
    '''
    各个条件缺失了多少条数值，共1460条数据
    Electrical          1
    MasVnrType          8
    MasVnrArea          8
    BsmtQual           37
    BsmtCond           37
    BsmtFinType1       37
    BsmtFinType2       38
    BsmtExposure       38
    GarageQual         81
    GarageFinish       81
    GarageYrBlt        81
    GarageType         81
    GarageCond         81
    LotFrontage       259
    FireplaceQu       690
    Fence            1179
    Alley            1369
    MiscFeature      1406
    PoolQC           1453
    :param datas:
    :return: datas

    '''
    #对于缺失值较多的那些特征用None填充
    for i in ['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond','LotFrontage',
    'FireplaceQu','Fence','Alley','MiscFeature','PoolQC','KitchenQual']:
        datas[i].fillna('None',inplace=True)
    #无配置的房屋物品，用0填充
    for i in ['BsmtUnfSF', 'TotalBsmtSF', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtHalfBath', \
         'MasVnrArea', 'GarageCars', 'GarageArea', 'GarageYrBlt']:
        datas[i].fillna(0,inplace=True)
    #少于40个缺省用众数填充
    for i in ['Electrical','MasVnrType','MasVnrArea','BsmtQual','BsmtCond',
    'BsmtFinType1','BsmtFinType2', 'BsmtExposure']:
        datas[i].fillna(datas[i].mode()[0],inplace=True)
    # print(datas.isnull().sum().sort_values(ascending=True))
    for i in ['MSSubClass', 'BsmtFullBath', 'BsmtHalfBath',
              'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
              'MoSold', 'YrSold', 'YearBuilt', 'YearRemodAdd',
              'LowQualFinSF', 'GarageYrBlt']:
        datas[i]=datas[i].astype(str)
    #对字符串的年份进行标签编码
    for i in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', "YrSold", 'MoSold']:
        datas[i]=LabelEncoder().fit_transform(datas[i])
    '''
        #对值有大小要求的不能标签编码的例如
    BsmtCond: Evaluates the general condition of the basement
       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
    '''
    # 对于None给与中评
    for i in ['BsmtCond', 'BsmtQual', 'ExterCond', 'ExterQual',
              'FireplaceQu', 'GarageCond', 'GarageQual',
              'HeatingQC','KitchenQual', 'PoolQC']:
        datas[i]=datas[i].map({'Po':1, 'Fa':2, 'None':3, 'TA':4,'Gd':5, 'Ex':6})

    datas['MSSubClass']=datas.MSSubClass.map({'180': 1, '30': 2, '45': 2,
                               '190': 3, '50': 3, '90': 3, '85': 4, '40': 4,
                               '160': 4, '70': 5, '20': 5, '75': 5, '80': 5, '150': 5, '120': 6, '60': 6})
    datas['Condition1'] = datas.Condition1.map({'Artery': 1,
                                                'Feedr': 2, 'RRAe': 2,'Norm': 3, 'RRAn': 3,
                                               'PosN': 4, 'RRNe': 4,'PosA': 5, 'RRNn': 5})
    datas['Condition2'] = datas.Condition2.map({'Artery': 1,
                                               'Feedr': 2, 'RRAe': 2, 'Norm': 3, 'RRAn': 3,
                                               'PosN': 4, 'RRNe': 4, 'PosA': 5, 'RRNn': 5})
    #对skew偏度较大的对数化处理，更加符合正态分布
    skewness=datas.select_dtypes(exclude=['object']).apply(lambda  x: skew(x.dropna()))
    skewness=skewness[abs(skewness)>1].index
    datas[skewness]=np.log1p(datas[skewness])
    datas=pd.get_dummies(datas)
    #取对数期间有可能出现极小值或者极大值，使用中位数填充
    cols=datas.columns.values.tolist()
    for col in cols:
        datas[col].values[np.isinf(datas[col].values)]=datas[col].median()
    # print(datas.isnull().sum())
    return datas

def datapca(train_data,datas):
    '''
    pac降维
    :param datas:
    :return:
    '''
    train_datas=datas[:train_data.shape[0]]
    test_datas=datas[train_data.shape[0]:]
    train_datas=RobustScaler().fit(train_datas).transform(train_datas)
    test_datas=RobustScaler().fit(test_datas).transform(test_datas)
    pca=PCA(n_components=400)
    train_datas=pca.fit_transform(train_datas)
    test_datas=pca.transform(test_datas)
    return train_datas,test_datas

class datastrain():
    '''
    SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
    MSSubClass: The building class
    MSZoning: The general zoning classification
    LotFrontage: Linear feet of street connected to property #1201 non-null
    LotArea: Lot size in square feet
    Street: Type of road access
    Alley: Type of alley access #91 non-null
    LotShape: General shape of property
    LandContour: Flatness of the property
    Utilities: Type of utilities available
    LotConfig: Lot configuration
    LandSlope: Slope of property
    Neighborhood: Physical locations within Ames city limits
    Condition1: Proximity to main road or railroad
    Condition2: Proximity to main road or railroad (if a second is present)
    BldgType: Type of dwelling
    HouseStyle: Style of dwelling
    OverallQual: Overall material and finish quality
    OverallCond: Overall condition rating
    YearBuilt: Original construction date
    YearRemodAdd: Remodel date
    RoofStyle: Type of roof
    RoofMatl: Roof material
    Exterior1st: Exterior covering on house
    Exterior2nd: Exterior covering on house (if more than one material)
    MasVnrType: Masonry veneer type #1452 non-null
    MasVnrArea: Masonry veneer area in square feet #1452 non-null
    ExterQual: Exterior material quality
    ExterCond: Present condition of the material on the exterior
    Foundation: Type of foundation
    BsmtQual: Height of the basement #1423 non-null
    BsmtCond: General condition of the basement #1423 non-null
    BsmtExposure: Walkout or garden level basement walls #1422 non-null
    BsmtFinType1: Quality of basement finished area #1423 non-null
    BsmtFinSF1: Type 1 finished square feet
    BsmtFinType2: Quality of second finished area (if present) #1422 non-null
    BsmtFinSF2: Type 2 finished square feet
    BsmtUnfSF: Unfinished square feet of basement area
    TotalBsmtSF: Total square feet of basement area
    Heating: Type of heating
    HeatingQC: Heating quality and condition
    CentralAir: Central air conditioning
    Electrical: Electrical system #1459 non-null
    1stFlrSF: First Floor square feet
    2ndFlrSF: Second floor square feet
    LowQualFinSF: Low quality finished square feet (all floors)
    GrLivArea: Above grade (ground) living area square feet
    BsmtFullBath: Basement full bathrooms
    BsmtHalfBath: Basement half bathrooms
    FullBath: Full bathrooms above grade
    HalfBath: Half baths above grade
    Bedroom: Number of bedrooms above basement level
    Kitchen: Number of kitchens
    KitchenQual: Kitchen quality
    TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
    Functional: Home functionality rating
    Fireplaces: Number of fireplaces
    FireplaceQu: Fireplace quality
    GarageType: Garage location
    GarageYrBlt: Year garage was built
    GarageFinish: Interior finish of the garage
    GarageCars: Size of garage in car capacity
    GarageArea: Size of garage in square feet
    GarageQual: Garage quality
    GarageCond: Garage condition
    PavedDrive: Paved driveway
    WoodDeckSF: Wood deck area in square feet
    OpenPorchSF: Open porch area in square feet
    EnclosedPorch: Enclosed porch area in square feet
    3SsnPorch: Three season porch area in square feet
    ScreenPorch: Screen porch area in square feet
    PoolArea: Pool area in square feet
    PoolQC: Pool quality
    Fence: Fence quality
    MiscFeature: Miscellaneous feature not covered in other categories
    MiscVal: $Value of miscellaneous feature
    MoSold: Month Sold
    YrSold: Year Sold
    SaleType: Type of sale
    SaleCondition: Condition of sale
    '''
    def __init__(self,model):
        self.model=model
        self.KF=KFold(n_splits=10,random_state=23,shuffle=True)
    def gradient_get(self,x,y,param_gradient):
        grid=GridSearchCV(self.model,param_gradient,cv=self.KF,
                          scoring='neg_mean_squared_error',n_jobs=2)
        grid.fit(x,y)
        print(grid.best_params_,np.sqrt(-grid.best_score_))
        grid.cv_results_['mean_test_score']=np.sqrt(-grid.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid.cv_results_)[['params','mean_test_score','std_test_score']])
        return grid

def rmse_cv(model,x,y):
    rmse=np.sqrt(-cross_val_score(model,x,y,cv=KFold(n_splits=10,random_state=23,shuffle=True)))
    return rmse
def main():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    train_y = train_data.iloc[:, -1]
    datas = train_data.append(test_data, ignore_index=True)

    datas.drop(['SalePrice'], axis=1, inplace=True)
    datas.drop(['Id'], axis=1, inplace=True)
    # train_data.info()
    # train_data.describe()
    #观察数据的缺省值
    # print(datas.isnull().sum().sort_values(ascending=True))
    # print(train_data['MSZoning'].mode())
    datas=data_value_deal(datas)
    train_data,test_data=datapca(train_data,datas)
    # print(train_data[:5],test_data[:5])
    model=XGBRegressor()
    grid=datastrain(model).gradient_get(train_data,train_y,{
        'max_depth':[8],
        'learning_rate':[0.01],
        'n_estimators':[10000]
    })
    model=grid.best_estimator_
    result = rmse_cv(model, train_data, train_y)
    cv_mean = result.mean()
    cv_std = result.std()
    print('cv_mean:', cv_mean, 'cv_std:', cv_std)
    prey=model.predict(train_data)
    model.save_model('001.model')
    acc=np.sqrt(np.power(prey-train_y,2))
    print(acc[:5],acc.sum())
    #对于离散型数据，转换类型
    # print(train_data.MSSubClass.head())


if __name__ == '__main__':
    main()

