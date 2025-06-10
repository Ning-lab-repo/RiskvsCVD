import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


RANDOM_SEED = 42  
np.random.seed(RANDOM_SEED)  

input_path = r"G:\CVD\3.KNN\allpeople-20836.xlsx"
data = pd.read_excel(input_path)

scaler = StandardScaler()
X = data.iloc[:, 2:59]  
X_standardized = scaler.fit_transform(X)  
y = data.iloc[:, -1]  


X_train, X_test, y_train, y_test = train_test_split(
    X_standardized, y, 
    test_size=0.3, 
    random_state=RANDOM_SEED,  
    stratify=y
)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

param_grid_rf = {
    'n_estimators': [200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

param_grid_svm = {
    'C': [0.1, 1.0, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

param_grid_lr = {
    'C': [0.1, 1.0, 10],
    'max_iter': [200, 300, 500],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs']
}

param_grid_xgb = {
    'n_estimators': [300, 500],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
    'lambda': [0, 0.1, 1, 10],  
    'alpha': [0, 0.1, 1, 10]    
}


rf = RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced')
svm = SVC(probability=True, random_state=RANDOM_SEED, class_weight='balanced')
lr = LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced')
xgb_model = xgb.XGBClassifier(
    random_state=RANDOM_SEED, 
    use_label_encoder=False, 
    eval_metric='mlogloss', 
    scale_pos_weight=class_weight_dict[0] / class_weight_dict[1]
)


cv = StratifiedKFold(
    n_splits=5, 
    shuffle=True, 
    random_state=RANDOM_SEED  
)

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')

grid_search_rf.fit(X_train, y_train)
grid_search_svm.fit(X_train, y_train)
grid_search_lr.fit(X_train, y_train)
grid_search_xgb.fit(X_train, y_train)



best_rf_model = grid_search_rf.best_estimator_
best_svm_model = grid_search_svm.best_estimator_
best_lr_model = grid_search_lr.best_estimator_
best_xgb_model = grid_search_xgb.best_estimator_


best_models = {
    'Model': ['Random Forest', 'SVM', 'Logistic Regression', 'XGBoost'],
    'Best Params': [
        grid_search_rf.best_params_,
        grid_search_svm.best_params_,
        grid_search_lr.best_params_,
        grid_search_xgb.best_params_
    ]
}


best_models_df = pd.DataFrame(best_models)


print(best_models_df)


output_path = "G:/CVD/5.ROC/1.风vsCVD/best_model.xlsx"  
best_models_df.to_excel(output_path, index=False)

print(f"save: {output_path}")


y_pred_rf_proba = best_rf_model.predict_proba(X_test)[:, 1]
y_pred_svm_proba = best_svm_model.predict_proba(X_test)[:, 1]
y_pred_lr_proba = best_lr_model.predict_proba(X_test)[:, 1]
y_pred_xgb_proba = best_xgb_model.predict_proba(X_test)[:, 1]


roc_auc_rf = roc_auc_score(y_test, y_pred_rf_proba)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm_proba)
roc_auc_lr = roc_auc_score(y_test, y_pred_lr_proba)
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb_proba)


fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_proba)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm_proba)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba)


plt.figure(figsize=(10, 8))

plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {roc_auc_rf:.4f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.4f})")
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC = {roc_auc_lr:.4f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_xgb:.4f})")


plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate", fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC Curve for Test Set", fontsize=18)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(False)

save_path = "G:/CVD/5.ROC/1.riskvsCVD/ROC.png" 
plt.savefig(save_path, dpi=300, bbox_inches='tight')  
print(f"ROC : {save_path}")

plt.show()
