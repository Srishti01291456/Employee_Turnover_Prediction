import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from data_preprocessing import (
    load_data,
    preprocess_data
)

# Load data
df = load_data("../data/employee_turnover.csv")

# Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Hyperparameter tuning
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate
predictions = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)

# Save model
pickle.dump(
    best_model,
    open("../models/turnover_model.pkl", "wb")
)

# Save scaler
pickle.dump(
    scaler,
    open("../models/scaler.pkl", "wb")
)

print("Model Saved Successfully")
