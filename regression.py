import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from classification import getTrainingData
from optims.best_classifier import best_model

def evaluate_features_and_regression():
    df, target = getTrainingData()
    classifier = best_model
    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=0.2, shuffle=True, random_state=42
    )
    classifier.fit(X_train, y_train)

    # Compute permutation importance for feature evaluation
    importance = permutation_importance(
        classifier, X_train, y_train, n_repeats=100, n_jobs=-1
    )

    print(f"Feature Importance: {importance.importances_mean}")

    feature_importances = importance.importances_mean
    
    # Extract the top 10 features
    top_features_idx = np.argsort(feature_importances)[-10:][::-1]
    top_features = X_train.columns[top_features_idx]

    print(f"Top 10 features: {top_features.tolist()}")

    # Create a subset
    X_train_top = X_train.iloc[:, top_features_idx]
    X_test_top = X_test.iloc[:, top_features_idx]

    # Train a linear regression model and evaluate
    regressor = LinearRegression()
    regressor.fit(X_train_top, y_train)

    y_pred = regressor.predict(X_test_top)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE) of regression: {mse}")

    # Print the top 10 features and regression coefficients
    results_df = pd.DataFrame({
        "Feature": top_features,
        "Coefficient": regressor.coef_
    })

    results_df.to_csv('./data/top_10_features_and_coefficients.csv', index=False)
    print("Top 10 features and regression coefficients saved to ./data/top_10_features_and_coefficients.csv")


if __name__ == "__main__":
    evaluate_features_and_regression()