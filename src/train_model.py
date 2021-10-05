"""
Inside kaggle_titanic_deploy folder, run: python src/train_model.py
"""
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from feature_engineering_pipeline import FeatureEngineering


def main():
    train = pd.read_csv('data/train.csv')
    X_train = train.drop(['Survived'], axis=1)
    y_train = train['Survived']

    X_test = pd.read_csv('data/test.csv')

    features = ['Pclass', 'Age', 'Sex', 'Fare']
    X_train = X_train[features]
    X_test = X_test[features]

    feature_engineering_pipeline = FeatureEngineering().get_pipeline()
    X_train = feature_engineering_pipeline.fit_transform(X_train)

    X_test = feature_engineering_pipeline.transform(X_test)

    X_train.to_csv('data/train_after_feature_engineering.csv', index=False)
    X_test.to_csv('data/test_after_feature_engineering.csv', index=False)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    with open('models/model.pkl', 'wb') as pickle_file:
        pickle.dump(model, pickle_file)


if __name__ == '__main__':
    main()
