from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle


class OneHotEncode(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature = 'Sex'

    def fit(self, df):
        df[self.feature] = df[self.feature].astype(str)
            
        enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary')

        enc.fit(df['Sex'].values.reshape(-1, 1))

        with open(f"models/ohe.pkl", 'wb') as ohe_filename:
            pickle.dump(enc, ohe_filename)

        return self

    def transform(self, df):

        df[self.feature] = df[self.feature].astype(str)

        with open("models/ohe.pkl", "rb") as pickle_file:
            enc = pickle.load(pickle_file)

        df = self.append_ohe_to_dataframe(enc, df)

        df = df.drop(columns=self.feature)
        
        return df

    def append_ohe_to_dataframe(self, enc, df):
        ohe_feature = enc.transform(df[self.feature].values.reshape(-1, 1)).toarray()
        df_ohe = pd.DataFrame(ohe_feature, columns=enc.get_feature_names_out([self.feature]))
        df_ohe.index = df.index

        return pd.concat([df, df_ohe], axis=1)
