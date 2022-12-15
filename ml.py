from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor, CatBoostClassifier
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from scipy.stats import mannwhitneyu
import seaborn as sns
import shap
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._base import _get_weights
from sklearn.utils.validation import check_array

class MedianKNNRegressor(KNeighborsRegressor):
    def predict(self, X, return_match_index=False):
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        ######## Begin modification
        if weights is None:
            y_pred = np.median(_y[neigh_ind], axis=1)
        else:
            # y_pred = weighted_median(_y[neigh_ind], weights, axis=1)
            raise NotImplementedError("weighted median")
        ######### End modification

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        if return_match_index:
            d = _y[neigh_ind]
            middle_idx = self.n_neighbors//2 #assumes n_neighbors is odd
            median_idx = np.argsort(d)[:, middle_idx]
            nearest_index = []
            for ni, mi in zip(neigh_ind, median_idx):
                nearest_index.append(ni[mi])

            nearest_matched_index = np.array(nearest_index)

            return y_pred, nearest_matched_index
        else:
            return y_pred



class FactorsCalculatorRegressor:
    def __init__(
        self,
        num_cols, 
        cat_cols, 
        target_col,
        data=None,
        train=None,
        test=None,
        random_st=42, 
        log_target=False,
        estimator='catboost',
        imputer_strategy='simple',
):
        self.num_cols, self.cat_cols = num_cols, cat_cols
        self.estimator = estimator
        self.target_col = target_col
        self.imputer_strategy = imputer_strategy
        if data is not None:
            self.data = data.copy()

            if log_target:
                self.data['target'] = self.data[target_col].apply(np.log1p)
            else:
                self.data['target'] = self.data[target_col]

            self.data.drop(columns=target_col, inplace=True)

            self.train, self.test = train_test_split(
                self.data, random_state=random_st
            )
        else:
            self.train = train.copy()
            self.train['target'] = self.train[target_col]
            self.train.drop(columns=target_col, inplace=True)
            self.test = test.copy()
            self.test['target'] = self.test[target_col]
            self.test.drop(columns=target_col, inplace=True)
            self.data = pd.concat([train, test])

        prep = self.get_pipeline_preprocessor()

        if estimator=='random_forest':
            self.reg = RandomForestRegressor(random_state=random_st)
        elif estimator=='catboost':
            self.reg = CatBoostRegressor(verbose=0)
        else:
            self.reg = LinearRegression()
        
        print('Using estimator', self.reg)
        
        self.model = self.get_model_pipeline(prep, self.reg)
    
    def fit_report(self):
        self.model.fit(self.train, self.train.target)
        print('R2 Score:', self.model.score(self.test, self.test.target))
        self.y_pred = self.model.predict(self.test)
        rmse = mean_squared_error(self.test.target, self.y_pred, squared=False)
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
        with_feature = self.data.loc[filter_feature, 'target'].values
        without_feature = self.data.loc[~filter_feature, 'target'].values
        sns.histplot(
            x="target", 
            hue=binary_column, 
            data=self.data,
            multiple="dodge",
            binwidth=1
            #element="step"
        ).set_title(
            'Distributions of binary feature'
        )
        plt.show()
        sns.boxplot(
            x=binary_column,
            y='target',
            data=self.data
        ).set_title(
            'Boxplot of binary feature'
        )
        plt.show()
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
        elif self.estimator=='random_forest' or self.estimator=='catboost':
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
    
    def get_shap(self, return_shap_values=False):
        explainer = shap.TreeExplainer(self.model['estimator'])
        observations = self.model['preprocessor'].transform(self.test)
        feature_names = self._proc_output_names()
        observations = pd.DataFrame(observations, columns=feature_names)
        shap_values = explainer(observations)
        shap.summary_plot(shap_values)
        if return_shap_values:
            return shap_values
    
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
                #('scaler', StandardScaler())
             
            ]
        )

        num_pipe = Pipeline(
            [
                ('imputer', self.num_out_transformer),
                #('scaler', StandardScaler()) # Needed for coeff outputting Z-Score
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

class FactorsCalculatorClassifier(object):
    def __init__(
        self,
        num_cols, 
        cat_cols, 
        target_col,
        data=None,
        train=None,
        test=None,
        random_st=42, 
        log_target=False,
        estimator='catboost',
        imputer_strategy='simple',
):
        self.num_cols, self.cat_cols = num_cols, cat_cols
        self.estimator = estimator
        self.target_col = target_col
        self.imputer_strategy = imputer_strategy
        if data is not None:
            self.data = data.copy()

            if log_target:
                self.data['target'] = self.data[target_col].apply(np.log1p)
            else:
                self.data['target'] = self.data[target_col]

            self.data.drop(columns=target_col, inplace=True)

            self.train, self.test = train_test_split(
                self.data, random_state=random_st
            )
        else:
            self.train = train.copy()
            self.train['target'] = self.train[target_col]
            self.train.drop(columns=target_col, inplace=True)
            self.test = test.copy()
            self.test['target'] = self.test[target_col]
            self.test.drop(columns=target_col, inplace=True)
            self.data = pd.concat([train, test])

        prep = self.get_pipeline_preprocessor()

        if estimator=='catboost':
            self.estimator = CatBoostClassifier(verbose=0)
        
        print('Using estimator', self.estimator)
        
        self.model = self.get_model_pipeline(prep, self.estimator)
    
    def fit_report(self):
        self.model.fit(self.train, self.train.target)
        print('Score:', self.model.score(self.test, self.test.target))
        self.y_pred = self.model.predict(self.test)
        print('Classification report')
        print(classification_report(self.test.target, self.y_pred))

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
        with_feature = self.data.loc[filter_feature, 'target'].values
        without_feature = self.data.loc[~filter_feature, 'target'].values
        sns.histplot(
            x="target", 
            hue=binary_column, 
            data=self.data,
            multiple="dodge",
            binwidth=1
            #element="step"
        ).set_title(
            'Distributions of binary feature'
        )
        plt.show()
        sns.boxplot(
            x=binary_column,
            y='target',
            data=self.data
        ).set_title(
            'Boxplot of binary feature'
        )
        plt.show()
        U1, p = mannwhitneyu(with_feature, without_feature, alternative='two-sided')
        print('Two-sided p-value for Mann–Whitney U test is', p)
        return p

    def get_coefficients(self):
        if self.estimator=='linear':
            output_names = self._proc_output_names()
            #coef = pd.DataFrame(
            #    self.estimator.coef_,
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
        elif self.estimator=='random_forest' or self.estimator=='catboost':
            output_names = self._proc_output_names()
            feat_imp = pd.DataFrame(
                {
                    'features': output_names,
                    'importance': self.estimator.feature_importances_
                }   
            )

            feat_imp = feat_imp.sort_values(
                by='importance',
                ascending=False
            )

            return feat_imp


    def get_feature_importances(self, plot=True):
        output_names = self._proc_output_names()
        feat_imp = pd.Series(
            self.estimator.feature_importances_,
            index=output_names
        )

        feat_imp = feat_imp.sort_values(ascending=False)

        return feat_imp
    
    def get_shap(self, return_shap_values=False):
        explainer = shap.TreeExplainer(self.model['estimator'])
        observations = self.model['preprocessor'].transform(self.test)
        feature_names = self._proc_output_names()
        observations = pd.DataFrame(observations, columns=feature_names)
        shap_values = explainer(observations)
        shap.summary_plot(shap_values, observations)
        if return_shap_values:
            return shap_values
    
    def plot_importance(
        self,
        feat_imp,
        graph_title="Model feature importance"
    ):
        sns.barplot(feat_imp).set(title=graph_title)
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
                    #min_frequency=0.05 # ignore minority categories
                )),
                #('scaler', StandardScaler())
             
            ]
        )

        num_pipe = Pipeline(
            [
                ('imputer', self.num_out_transformer),
                #('scaler', StandardScaler()) # Needed for coeff outputting Z-Score
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
