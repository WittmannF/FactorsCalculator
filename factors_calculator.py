from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd

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
