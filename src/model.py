import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

#load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

#save test ids
test_ids = test["id"].copy()
train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)

#encode target
le = LabelEncoder()
train["Status"] = le.fit_transform(train["Status"])

y = train["Status"]
X = train.drop(columns=["Status"])



#process categorical features
cat_cols = X.select_dtypes(include=["string", "object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"\nCategorical columns: {cat_cols}")
print(f"Numerical columns: {num_cols}")

#OneHotEncoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(X[cat_cols])

X_cat_encoded = encoder.transform(X[cat_cols])
test_cat_encoded = encoder.transform(test[cat_cols])

feature_names = encoder.get_feature_names_out(cat_cols)
X_cat_df = pd.DataFrame(X_cat_encoded, columns=feature_names, index=X.index)
test_cat_df = pd.DataFrame(test_cat_encoded, columns=feature_names, index=test.index)

#combine features
if num_cols:
    X_final = pd.concat([X[num_cols].reset_index(drop=True), X_cat_df.reset_index(drop=True)], axis=1)
    test_final = pd.concat([test[num_cols].reset_index(drop=True), test_cat_df.reset_index(drop=True)], axis=1)
else:
    X_final = X_cat_df.reset_index(drop=True)
    test_final = test_cat_df.reset_index(drop=True)

print(f"\nFinal training shape: {X_final.shape}")
print(f"Final test shape: {test_final.shape}")



#model parameters
params = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 100,
    "subsample": 0.6,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "min_split_gain": 0.05,
    "min_child_weight": 5,
    "n_estimators": 1500,
    "random_state": 42,
    "verbose": -1
}



#train ensemble
print("\n" + "="*60)
print("TRAINING ENSEMBLE")
print("="*60)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
n_ensemble = 7
ensemble_preds = np.zeros((len(test_final), 3))

for i in range(n_ensemble):
    print(f"\nTraining ensemble model {i+1}/{n_ensemble}")
    
    #vary parameters for diversity
    params_ensemble = params.copy()
    params_ensemble['random_state'] = 42 + i * 10
    params_ensemble['learning_rate'] = 0.01 * (1 + 0.1 * (i % 3 - 1))
    
    model_preds = np.zeros((len(test_final), 3))
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_final, y)):
        X_train, X_valid = X_final.iloc[train_idx], X_final.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = lgb.LGBMClassifier(**params_ensemble)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(50)]
        )
        
        model_preds += model.predict_proba(test_final) / n_splits
    
    ensemble_preds += model_preds / n_ensemble

#calibrate probabilities
ensemble_preds = ensemble_preds / ensemble_preds.sum(axis=1, keepdims=True)



#create submission
print("\n" + "="*60)
print("GENERATING SUBMISSION")
print("="*60)

class_names = le.classes_
submission = pd.DataFrame({
    "id": test_ids,
    class_names[0]: np.round(ensemble_preds[:, 0], 6),
    class_names[1]: np.round(ensemble_preds[:, 1], 6),
    class_names[2]: np.round(ensemble_preds[:, 2], 6),
})

submission.to_csv("submission.csv", index=False, float_format='%.6f')

print(f"\nSubmission file created: submission.csv")
print(f"Shape: {submission.shape}")
print("\nFirst 5 rows:")
print(submission.head())
