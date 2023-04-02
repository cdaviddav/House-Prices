import pandas as pd
import numpy as np
from scipy.stats import skew

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

from typing import Optional




class ImputMissingValuesNumeric(BaseEstimator, TransformerMixin):

    def __init__(self, imputer_function:str):
        self.imputer_function = imputer_function
        self.imputer_num = None
        self.num_features = None

    def fit(self, x:pd.DataFrame, y:Optional[pd.DataFrame]=None) -> "ImputMissingValuesNumeric":
        if self.imputer_function == "SimpleImputer":
            imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        elif self.imputer_function == "KNNImputer":
            imputer = KNNImputer(missing_values=np.nan)
        
        self.num_features = list(x.describe())
        self.imputer_num = imputer.fit(x[self.num_features])
        return self

    def transform(self, x:pd.DataFrame) -> pd.DataFrame:
        try:
            x[self.imputer_num.feature_names_in_] = self.imputer_num.transform(x[self.imputer_num.feature_names_in_])
        except KeyError:
            pass
        return x


class ImputMissingValuesCategoric(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.imputer_cat = None
        self.cat_features = None

    def fit(self, x:pd.DataFrame, y:Optional[pd.DataFrame]=None) -> "ImputMissingValuesCategoric":
        imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        
        self.cat_features = list(x.describe(include=['O']))
        self.imputer_cat = imputer.fit(x[self.cat_features])
        return self

    def transform(self, x:pd.DataFrame) -> pd.DataFrame:
        try:
            x[self.imputer_cat.feature_names_in_] = self.imputer_cat.transform(x[self.imputer_cat.feature_names_in_])
        except KeyError:
            pass
        return x


def find_season(month, hemisphere):
    if hemisphere == 'Southern':
        season_month_south = {
            12:'Summer', 1:'Summer', 2:'Summer',
            3:'Autumn', 4:'Autumn', 5:'Autumn',
            6:'Winter', 7:'Winter', 8:'Winter',
            9:'Spring', 10:'Spring', 11:'Spring'}
        return season_month_south.get(month)
        
    elif hemisphere == 'Northern':
        season_month_north = {
            12:'Winter', 1:'Winter', 2:'Winter',
            3:'Spring', 4:'Spring', 5:'Spring',
            6:'Summer', 7:'Summer', 8:'Summer',
            9:'Autumn', 10:'Autumn', 11:'Autumn'}
        return season_month_north.get(month)
    else:
        print('Invalid selection. Please select a hemisphere and try again')


class CreateNewFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, x:pd.DataFrame, y:Optional[pd.DataFrame]=None) -> "CreateNewFeatures":
        self.x_train = x.copy()
        return self

    def transform(self, x:pd.DataFrame) -> pd.DataFrame:
        x["TotalFS"] = x["1stFlrSF"] + x["2ndFlrSF"] + x["GrLivArea"]
        x["MeanSFRoom"] = round(x["TotalFS"] / x["TotRmsAbvGrd"], 0)
        x["YearsBeforeWork"] = x["YearRemodAdd"] - x["YearBuilt"]
        x["TotalBath"] = x["FullBath"] + 0.5*x["HalfBath"] + x["BsmtFullBath"] + 0.5*x["BsmtHalfBath"]
        x["TotalFS_TotalBath"] = x["TotalFS"] / x["TotalBath"]
        x["GarageArea_GarageCars"] = x["GarageArea"] / x["GarageCars"]
        x["TotalPorchSF"] = x["OpenPorchSF"] + x["EnclosedPorch"] + x["3SsnPorch"] + x["ScreenPorch"]
        x["YearsBeforeSold"] = x["YrSold"] - x["YearBuilt"]
        x["SeasonSold"] = x["MoSold"].apply(lambda x: find_season(x, "Northern"))

        x['PoolArea_bin'] = x['PoolArea'].apply(lambda x: 1 if x>0 else 0)
        x['TotalPorchSF_bin'] = x['TotalPorchSF'].apply(lambda x: 1 if x>0 else 0)
        x['GarageArea_bin'] = x['GarageArea'].apply(lambda x: 1 if x>0 else 0)
        x['MiscVal_bin'] = x['MiscVal'].apply(lambda x: 1 if x>0 else 0)

        x.replace([np.inf, -np.inf], 0, inplace=True)
        return x



def compute_skewed_features(df):
    """
    compute the skewness of all numeric features and the total number of unique values
    return only the features that have a relevant skewness
    """
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    skewed_feats = pd.DataFrame(index=numeric_feats, columns=['skewness', 'unique_values'])
    skewed_feats['skewness'] = df[numeric_feats].apply(lambda x: skew(x))
    skewed_feats['unique_values'] = df.nunique()
    skewed_feats['percentage_0'] = df[df == 0].count(axis=0)/len(df.index)
    skewed_feats = skewed_feats[
        ((skewed_feats['skewness'] > 3) | (skewed_feats['skewness'] < -3)) & 
        (skewed_feats['unique_values'] > 10) &
        (skewed_feats['percentage_0'] < 0.5)
        ]

    return skewed_feats


class SkewedFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transform_skewed_features_flag:bool):
        self.transform_skewed_features_flag = transform_skewed_features_flag
        self.skewed_features = None

    def fit(self, x:pd.DataFrame, y:Optional[pd.DataFrame]=None) -> "SkewedFeatureTransformer":
        df_skewed_features = compute_skewed_features(x)
        self.skewed_features = df_skewed_features.index
        return self

    def transform(self, x:pd.DataFrame) -> pd.DataFrame:
        if self.transform_skewed_features_flag==True:
            for feature in list(self.skewed_features):
                x[feature] = x[feature].apply(np.log)
                # the not transformed data that contains 0
                # after the transformation we have -inf values that have to be replaced by 0
                x[feature][np.isneginf(x[feature])]=0

        return x


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, x:pd.DataFrame, y:Optional[pd.DataFrame]=None) -> "LabelEncoderTransformer":
        self.x_train = x.copy()
        return self

    def transform(self, x:pd.DataFrame) -> pd.DataFrame:
        
        map_Utilities = {
            'ELO': 0,
            'NoSeWa': 1,
            'NoSewr': 2,
            'AllPub': 3
        }
        x["Utilities"] = x["Utilities"].map(map_Utilities)


        map_LandSlope = {
            'Gtl': 0,
            'Mod': 1,
            'Sev': 2
        }
        x["LandSlope"] = x["LandSlope"].map(map_LandSlope)
        

        map_HouseStyle = {
            '1Story': 0,
            '1.5Fin': 1,
            '1.5Unf': 2,
            '2Story': 3,
            '2.5Fin': 4,
            '2.5Unf': 5,
            'SFoyer': 6,
            'SLvl': 7
        }
        x["HouseStyle"] = x["HouseStyle"].map(map_HouseStyle)


        map_ExterQual = {
            'Po': 0,
            'Fa': 1,
            'TA': 2,
            'Gd': 3,
            'Ex': 4
        }
        x["ExterQual"] = x["ExterQual"].map(map_ExterQual)
        x["ExterCond"] = x["ExterCond"].map(map_ExterQual)
        

        map_BsmtQual = {
            'NA': 0,
            'Po': 1,
            'Fa': 2,
            'TA': 3,
            'Gd': 4,
            'Ex': 5
        }
        x["BsmtQual"] = x["BsmtQual"].map(map_BsmtQual)
        x["BsmtCond"] = x["BsmtCond"].map(map_BsmtQual)
        x["GarageQual"] = x["GarageQual"].map(map_BsmtQual)
        x["GarageCond"] = x["GarageCond"].map(map_BsmtQual)


        map_BsmtExposure = {
            'NA': 0,
            'No': 1,
            'Mn': 2,
            'Av': 3,
            'Gd': 4
        }
        x["BsmtExposure"] = x["BsmtExposure"].map(map_BsmtExposure)


        map_BsmtFinType1 = {
            'NA': 0,
            'Unf': 1,
            'LwQ': 2,
            'Rec': 3,
            'BLQ': 4,
            'ALQ': 5,
            'GLQ': 6
        }
        x["BsmtFinType1"] = x["BsmtFinType1"].map(map_BsmtFinType1)
        x["BsmtFinType2"] = x["BsmtFinType2"].map(map_BsmtFinType1)


        map_HeatingQC = {
            'Po': 0,
            'Fa': 1,
            'TA': 2,
            'Gd': 3,
            'Ex': 4
        }
        x["HeatingQC"] = x["HeatingQC"].map(map_HeatingQC)
        x["KitchenQual"] = x["KitchenQual"].map(map_HeatingQC)


        map_Functional = {
            'Sal': 0,
            'Sev': 1,
            'Maj2': 2,
            'Maj1': 3,
            'Mod': 4,
            'Min2': 5,
            'Min1': 6,
            'Typ': 7
        }
        x["Functional"] = x["Functional"].map(map_Functional)


        map_GarageFinish = {
            'NA': 0,
            'Unf': 1,
            'RFn': 2,
            'Fin': 3
        }
        x["GarageFinish"] = x["GarageFinish"].map(map_GarageFinish)


        map_PavedDrive = {
            'Y': 0,
            'P': 1,
            'N': 2
        }
        x["PavedDrive"] = x["PavedDrive"].map(map_PavedDrive)


        map_SeasonSold = {
            'Winter': 0,
            'Spring': 1,
            'Summer': 2,
            'Autumn': 3
        }
        x["SeasonSold"] = x["SeasonSold"].map(map_SeasonSold)


        return x


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, ohe_min_frequency:float, ohe_max_categories:int):
        self.ohe_min_frequency = ohe_min_frequency
        self.ohe_max_categories = ohe_max_categories
        self.cat_vars=None
        self.enc=None

    def fit(self, x:pd.DataFrame, y:Optional[pd.DataFrame]=None) -> "OneHotEncoderTransformer":
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False, drop="if_binary",
            min_frequency=self.ohe_min_frequency, max_categories=self.ohe_max_categories)
        
        self.cat_vars = x.dtypes[x.dtypes == "object"].index
        self.enc = enc.fit(x[self.cat_vars])
        return self

    
    def transform(self, x:pd.DataFrame) -> pd.DataFrame:
        ohe = pd.DataFrame(self.enc.transform(x[self.cat_vars]), columns=self.enc.get_feature_names_out())
        
        x.reset_index(drop=True, inplace=True)
        ohe.reset_index(drop=True, inplace=True)

        x = pd.concat([x, ohe], axis=1).drop(self.cat_vars, axis=1)
        return x
    

class LowVarianceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variance_threshold:float):
        self.variance_threshold = variance_threshold
        self.sel_features=None
        self.sel=None
        self.sel_features_reduced=None

    def fit(self, x:pd.DataFrame, y:Optional[pd.DataFrame]=None) -> "LowVarianceTransformer":
        # remove all features that are either one or zero in more than 95% of the samples
        sel = VarianceThreshold(threshold=(self.variance_threshold * (1 - self.variance_threshold)))
        self.sel_features = list(x)
        # fit the VarianceThreshold object to the training data
        self.sel = sel.fit(x[self.sel_features])

        # get the column names after the variance threshold reduction
        self.sel_features_reduced = [self.sel_features[i] for i in self.sel.get_support(indices=True)]
        return self


    def transform(self, x:pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(self.sel.transform(x[self.sel_features]), columns=self.sel_features_reduced)
        return x


class CorrelationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, correlation_threshold:float):
        self.correlation_threshold = correlation_threshold
        self.to_drop=None

    def fit(self, x:pd.DataFrame, y:Optional[pd.DataFrame]=None) -> "CorrelationTransformer":
        corr_matrix = x.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find features with correlation higher than 0.9 or lower -0.9
        self.to_drop = [column for column in upper.columns if any((upper[column] > self.correlation_threshold) | (upper[column] < -self.correlation_threshold))]
        return self


    def transform(self, x:pd.DataFrame) -> pd.DataFrame:
        x =  x.drop(self.to_drop, axis=1)
        return x


class ScalerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columnprep__transformers_num) -> None:
        self.columnprep__transformers_num = columnprep__transformers_num
        self.transformer_not_num=None
        self.transformer_num=None

        if columnprep__transformers_num == "StandardScaler":
            self.scaler = StandardScaler()
        elif columnprep__transformers_num == "MinMaxScaler":
            self.scaler = MinMaxScaler()

    def fit(self, x:pd.DataFrame, y:Optional[pd.DataFrame]=None) -> "ScalerTransformer":
        self.transformer_not_num = [col for col in list(x) if (col.startswith("x") & col[1].isnumeric())]
        self.transformer_num = [col for col in list(x) if col not in self.transformer_not_num]
        
        self.scaler.fit(x[self.transformer_num], y)
        return self


    def transform(self, x:pd.DataFrame) -> pd.DataFrame:
        x_transform = self.scaler.transform(x[self.transformer_num])
        x_transform = pd.DataFrame(x_transform, index=x.index, columns=x[self.transformer_num].columns)
        return pd.concat([x_transform, x[self.transformer_not_num]], axis=1)


    def inverse_transform(self, x:pd.DataFrame) -> pd.DataFrame:
        x_inverse_transform = self.scaler.inverse_transform(x[self.transformer_num])
        x_inverse_transform = pd.DataFrame(x_inverse_transform, index=x.index, columns=x[self.transformer_num].columns)
        return pd.concat([x_inverse_transform, x[self.transformer_not_num]], axis=1)


