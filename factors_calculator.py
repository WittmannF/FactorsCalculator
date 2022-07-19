from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from scipy.stats import mannwhitneyu
import seaborn as sns
import shap
import pandas as pd
import numpy as np



class FactorsCalculatorRegressor:
    def __init__(
        self,
        data,
        num_cols, 
        cat_cols, 
        target_col, 
        random_st=42, 
        log_target=False,
        estimator='random_forest',
        imputer_strategy='simple',
):
        self.num_cols, self.cat_cols = num_cols, cat_cols
        self.estimator = estimator
        self.target_col = target_col
        self.imputer_strategy = imputer_strategy
        self.data = data.copy()

        if log_target:
            self.data['target'] = self.data[target_col].apply(np.log1p)
        else:
            self.data['target'] = self.data[target_col]

        self.data.drop(columns=target_col, inplace=True)

        self.train, self.test = train_test_split(
            self.data, random_state=random_st
        )

        prep = self.get_pipeline_preprocessor()

        if estimator=='random_forest':
            self.reg = RandomForestRegressor(random_state=random_st)
        else:
            self.reg = LinearRegression()
        
        print('Using estimator', self.reg)
        
        self.model = self.get_model_pipeline(prep, self.reg)
    
    def fit_report(self):
        self.model.fit(self.train, self.train.target)
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
    
    def test_binary_column(self, binary_column):
        print('Significance test on', binary_column)
        filter_feature = self.data[binary_column]==1
        with_feature = self.data.loc[filter_feature, self.target_col].values
        without_feature = self.data.loc[~filter_feature, self.target_col].values
        U1, p = mannwhitneyu(with_feature, without_feature, alternative='two-sided')
        print('Two-sided p-value for Mann–Whitney U test is', p)
        return p

    def get_coefficients(self):
        if self.estimator=='linear':
            output_names = self._proc_output_names()
            #coef = pd.DataFrame(
            #    self.reg.coef_,
            #    index=output_names
            #)

            coef = pd.DataFrame(
                {
                    'features': output_names,
                    'importance':self.clf.coef_
                }
            )

            sorted_idx = coef.apply(
                lambda x: abs(sum(x)), axis=1
            ).sort_values(ascending=False).index

            coef = coef.loc[sorted_idx]

            return coef
        elif self.estimator=='random_forest':
            output_names = self._proc_output_names()
            feat_imp = pd.DataFrame(
                {
                    'features': output_names,
                    'importance': self.reg.feature_importances_
                }   
            )

            feat_imp = feat_imp.sort_values(
                by='importance',
                ascending=False
            )

            return feat_imp


    def get_feature_importances(self):
        output_names = self._proc_output_names()
        feat_imp = pd.Series(
            self.reg.feature_importances_,
            index=output_names
        )

        feat_imp = feat_imp.sort_values(ascending=False)

        return feat_imp
    
    def get_shap(self):
        feature_names = self.data.columns

        def model_predict(data_asarray):
            data_asframe = pd.DataFrame(data_asarray, columns=feature_names)
            return self.model.predict(data_asframe)
        
        shap_kernel_explainer = shap.KernelExplainer(model_predict, self.train)

        shap_values = shap_kernel_explainer.shap_values(self.test)

        shap.summary_plot(shap_values, self.test)
    
    def plot_importance(
        self,
        feat_imp,
        graph_title="Model feature importance"
    ):
        sns.barplot(
            data=feat_imp,
            x='importance',
            y='features').set(title=graph_title)
        return

    def get_pipeline_preprocessor(self):
        if self.imputer_strategy=='knn':
            self.num_out_transformer = KNNImputer(n_neighbors=3)
        elif self.imputer_strategy=='simple':
            self.num_out_transformer = SimpleImputer(strategy='median')
        print('Imputer strategy:', self.num_out_transformer)

        cat_pipe = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='constant')),
                ('one_hot', OneHotEncoder(
                    sparse=False,
                    handle_unknown='infrequent_if_exist',
                    min_frequency=0.05 # ignore minority categories
                )),
                ('scaler', StandardScaler())
             
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

    def get_model_pipeline(self, preprocessor, reg):
        model = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('estimator', reg)
            ]
        )
        
        return model

class FactorsCalculator:
    def __init__(
        self, data, num_cols, cat_cols, target_col, random_st=42, 
        factorize_target=True
    ):
        self.num_cols, self.cat_cols = num_cols, cat_cols
        self.target_col = target_col

        if factorize_target:
            target, idx_to_label = data[target_col].factorize()
            data['target'] = target
            self.idx_to_label = idx_to_label
        else:
            data['target'] = data[target_col]
            self.idx_to_label = [target_col]



        self.train, self.test = train_test_split(
            data, random_state=random_st
        )
        prep = self.get_pipeline_preprocessor()
        self.clf = LogisticRegression()
        self.model = self.get_model_pipeline(prep, self.clf)
    
    def fit_report(self):
        self.model.fit(self.train, self.train[self.target_col])
        print('Score:', self.model.score(self.test, self.test.target))
        y_pred = self.model.predict(self.test)
        print(classification_report(self.test.target, y_pred))

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

    def get_coefficients(self):
        output_names = self._proc_output_names()
        coef = pd.DataFrame(
            self.clf.coef_,
            columns=output_names,
            #index=self.idx_to_label
        ).T

        sorted_idx = coef.apply(
            lambda x: abs(sum(x)), axis=1
        ).sort_values(ascending=False).index

        coef = coef.loc[sorted_idx]

        return coef
    
    def get_shap(self):
        pass
    
    def report_top_factors(self):
        pass

    def get_pipeline_preprocessor(self):
        self.num_out_transformer = SimpleImputer(strategy='median')

        cat_pipe = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='constant')),
                ('one_hot', OneHotEncoder(
                    sparse=False,
                    handle_unknown='infrequent_if_exist',
                    min_frequency=0.05 # ignore minority categories
                )),
                ('scaler', StandardScaler())
             
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
