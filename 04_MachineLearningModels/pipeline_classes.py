import pandas as pd
import numpy as np
from scipy.stats import skew

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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


def compute_skewed_features(df, skewed_threshold):
  """
  compute the skewness of all numeric features and the total number of unique values
  return only the features that have a relevant skewness
  """
  numeric_feats = df.dtypes[df.dtypes != "object"].index
  skewed_feats = pd.DataFrame(index=numeric_feats, columns=['skewness', 'unique_values'])
  skewed_feats['skewness'] = df[numeric_feats].skew()
  skewed_feats['unique_values'] = df.nunique()
  skewed_feats['percentage_0'] = df[df == 0].count(axis=0)/len(df.index)
  skewed_feats = skewed_feats[
      ((skewed_feats['skewness'] > skewed_threshold) | (skewed_feats['skewness'] < - skewed_threshold)) & 
      (skewed_feats['unique_values'] > 10) &
      (skewed_feats['percentage_0'] < 0.5)
      ]

  return skewed_feats


class SkewedFeatureTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, skewed_threshold:float):
    self.skewed_threshold = skewed_threshold
    self.features_to_transform = None
    self.features_to_transform_log1p = None
    self.features_to_transform_sqrt = None

  def fit(self, x:pd.DataFrame, y:Optional[pd.DataFrame]=None) -> "SkewedFeatureTransformer":
    self.features_to_transform = compute_skewed_features(x, self.skewed_threshold).index

    df_tmp = pd.DataFrame(index=self.features_to_transform, columns=["log1p", "sqrt"])
    df_tmp["log1p"] = x[self.features_to_transform].apply(np.log1p).skew()
    df_tmp["sqrt"] = x[self.features_to_transform].apply(np.sqrt).skew()

    # use idxmin(axis=1) to get the transformer with the lowest skewness
    df_tmp["transformer_skew"] = df_tmp.idxmin(axis=1)

    # create a list for each transformation that is applied in the transform function
    self.features_to_transform_log1p = df_tmp[df_tmp.transformer_skew == "log1p"].index
    self.features_to_transform_sqrt = df_tmp[df_tmp.transformer_skew == "sqrt"].index
    return self

  def transform(self, x:pd.DataFrame) -> pd.DataFrame:
      # use transformation based on lowest skew
      x[self.features_to_transform_log1p] = x[self.features_to_transform_log1p].apply(np.log1p)
      x[self.features_to_transform_sqrt] = x[self.features_to_transform_sqrt].apply(np.sqrt)
      return x


class CorrelationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, correlation_threshold:float):
        self.correlation_threshold = correlation_threshold
        self.to_drop=None

    def fit(self, x:np.ndarray, y:Optional[pd.DataFrame]=None) -> "CorrelationTransformer":
        corr_matrix = pd.DataFrame(x).corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation higher or lower than correlation_threshold
        self.to_drop = [column for column in upper.columns if any((upper[column] > self.correlation_threshold) | (upper[column] < -self.correlation_threshold))]
        return self

    def transform(self, x:np.ndarray) -> pd.DataFrame:
        x = pd.DataFrame(x).drop(self.to_drop, axis=1)
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


