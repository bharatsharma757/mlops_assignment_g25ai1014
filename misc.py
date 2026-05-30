import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import ssl

def load_data():
    ssl._create_default_https_context = ssl._create_unverified_context

    data_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    return pd.read_csv(data_url)


def split_data(df):
    X = df.drop("medv", axis=1)
    y = df["medv"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    return mse, r2
