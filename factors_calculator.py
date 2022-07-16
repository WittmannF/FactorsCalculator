import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


class FactorsCalculator:
  def __init__(self, data, num_cols, cat_cols,
               target_col, algorithm='forest',
               imputer_strategy='knn',
               random_st=42, log_target=False):
      df = data.copy()
      self.num_cols, self.cat_cols = num_cols, cat_cols
      self.target_col = target_col
      self.imputer_strategy = imputer_strategy
      self.algorithm = algorithm
      
      if log_target:
          df['target'] = data[target_col].apply(np.log1p)
      else:
          df['target'] = data[target_col]
        
      df.drop(columns='answer', inplace=True)
      
      if algorithm =='forest':
        self.clf = RandomForestRegressor(random_state=random_st)
        print('running random forest regressor\n')
      elif algorithm =='linear':
        self.clf = LinearRegression()
        print('running linear regression \n')

      self.train, self.test = train_test_split(
                              df, random_state=random_st)
      
      prep = self.get_pipeline_preprocessor()
      
      self.model = self.get_model_pipeline(prep, self.clf)

  def get_pipeline_preprocessor(self):
    if self.imputer_strategy=='knn':
      self.num_out_transformer = KNNImputer(n_neighbors=3)
      print('running KNNImputer \n')
    elif self.imputer_strategy=='simple':
      self.num_out_transformer = SimpleImputer(strategy='median')
      print('running SimpleImputer \n')
      
    cat_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='constant')),
            ('one_hot', OneHotEncoder(sparse=False,handle_unknown='ignore')),
                                      #min_frequency=0.05)
            #('scaler', StandardScaler())
        ]
    )
    num_pipe = Pipeline(
        [
            ('imputer', self.num_out_transformer),
            ('scaler', StandardScaler()) # Needed for coeff outputting Z-Score
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_pipe, self.cat_cols),
            ('num', num_pipe, self.num_cols)
        ]
    )  
    return preprocessor

  def get_model_pipeline(self, preprocessor, clf):
      model = Pipeline(
          steps=[
              ('preprocessor', preprocessor),
              ('estimator', clf)
          ]
      )
      return model

  def fit_report(self):
      self.model.fit(self.train, self.train['target'])
      print('R2 Score:', self.model.score(self.test, self.test.target))
      y_pred = self.model.predict(self.test)
      rmse = mean_squared_error(self.test.target, y_pred, squared=False)
      print('RMSE:', rmse)

  def _proc_output_names(self):
      model = self.model
      ct = model.steps[0][1]
      ct_cat = ct.transformers_[0][1]
      ct_cat_oh = ct_cat.steps[1][1]
      cat_out_names_ = list(ct_cat_oh.get_feature_names_out())
      for i, cat in enumerate(self.cat_cols):
          prefix = f'x{i}'
          for j, name in enumerate(cat_out_names_):
              if prefix in name[:len(prefix)]:
                  cat_out_names_[j] = name.replace(prefix, cat)
      
      self.cat_out_names_ = cat_out_names_
      ct_num = ct.transformers_[1][1]
      ct_num_si = ct_num.steps[0][1]
      self.num_out_names_ = list(ct_num_si.feature_names_in_)
      return self.cat_out_names_ + self.num_out_names_

  def get_coefficients(self, asc=False, return_rows=5):
      if self.algorithm =='linear':
        print('Getting linear coefficients...')
        output_names = self._proc_output_names()

        df_map = {'features': output_names,
                  'importance':self.clf.coef_}
        coef = pd.DataFrame(df_map)
        coef.sort_values(by='importance', 
                         ascending=asc, 
                         inplace=True)

        coef = coef[coef.features != "target"]
        rows = coef.shape
        if return_rows > rows[0]:
          print(f'printing head \n')
          coef = coef.head()
        else:
          print(f'printing {return_rows} rows\n')
          coef = coef.head(return_rows)
 
      elif self.algorithm =='forest':
        print('Getting forest coefficients...')
        output_names = self._proc_output_names()
        df_map = {'features': output_names,
                  'importance':self.clf.feature_importances_}
        coef = pd.DataFrame(df_map)
        coef.sort_values(by='importance', 
                         ascending=asc, 
                         inplace=True)
        
        coef = coef[coef.features != "target"]
        rows = coef.shape[0]
        if return_rows > rows:
          print(f'printing head \n')
          coef = coef.head()
        else:
          print(f'printing {return_rows} rows\n')
          coef = coef.head(return_rows)

      return coef

  def plot_importance(self, coef_df, graph_title="Model feature importance"):
    sns.barplot(data=coeffs,x='importance',
                y='features').set(title=graph_title)
    return 
