import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

input_path = r"G:\CVD\4.feature\20836_bio.xlsx"
data = pd.read_excel(input_path)

data['date'] = pd.to_datetime(data['date'], errors='coerce')

scaler = StandardScaler()
X = data.iloc[:, 2:35]  
X_standardized = scaler.fit_transform(X)  
y = data.iloc[:, -1]    

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42, stratify=y)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

rf_params = {'criterion': 'entropy', 
             'max_depth': 8, 
             'min_samples_leaf': 2, 
             'min_samples_split': 2, 
             'n_estimators': 50}
svm_params = {'C': 1.0, 
              'gamma': 'auto', 
              'kernel': 'rbf'}
lr_params = {'C': 0.1, 
             'max_iter': 200, 
             'penalty': 'l2', 
             'solver': 'liblinear'}
xgb_params = {
    'alpha': 1.5,  
    'lambda': 1.5, 
    'gamma': 0.2,  
    'colsample_bytree': 0.75, 
    'subsample': 0.85,  
    'learning_rate': 0.007,  
    'max_depth': 9,  
    'n_estimators': 5200,  
    'min_child_weight': 3,  
}


rf = RandomForestClassifier(**rf_params, random_state=42, class_weight='balanced')
svm = SVC(**svm_params, probability=True, random_state=42, class_weight='balanced')
lr = LogisticRegression(**lr_params, random_state=42, class_weight='balanced')
xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, use_label_encoder=False, eval_metric='mlogloss', scale_pos_weight=class_weight_dict[0] / class_weight_dict[1])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc_rf = []
cv_auc_svm = []
cv_auc_lr = []
cv_auc_xgb = []

for train_idx, val_idx in cv.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx] 

 
    rf.fit(X_train_fold, y_train_fold)
    svm.fit(X_train_fold, y_train_fold)
    lr.fit(X_train_fold, y_train_fold)
    xgb_model.fit(X_train_fold, y_train_fold)

  
    y_pred_rf_proba = rf.predict_proba(X_val_fold)[:, 1]
    y_pred_svm_proba = svm.predict_proba(X_val_fold)[:, 1]
    y_pred_lr_proba = lr.predict_proba(X_val_fold)[:, 1]
    y_pred_xgb_proba = xgb_model.predict_proba(X_val_fold)[:, 1]

    cv_auc_rf.append(roc_auc_score(y_val_fold, y_pred_rf_proba))
    cv_auc_svm.append(roc_auc_score(y_val_fold, y_pred_svm_proba))
    cv_auc_lr.append(roc_auc_score(y_val_fold, y_pred_lr_proba))
    cv_auc_xgb.append(roc_auc_score(y_val_fold, y_pred_xgb_proba))

print(f"Average AUC for Random Forest: {np.mean(cv_auc_rf):.4f}")
print(f"Average AUC for SVM: {np.mean(cv_auc_svm):.4f}")
print(f"Average AUC for Logistic Regression: {np.mean(cv_auc_lr):.4f}")
print(f"Average AUC for XGBoost: {np.mean(cv_auc_xgb):.4f}")


rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
lr.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

y_pred_rf_proba = rf.predict_proba(X_test)[:, 1]
y_pred_svm_proba = svm.predict_proba(X_test)[:, 1]
y_pred_lr_proba = lr.predict_proba(X_test)[:, 1]
y_pred_xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

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
plt.xlabel("False Positive Rate", fontsize=25)
plt.ylabel("True Positive Rate", fontsize=25)
plt.title("Biochemical detection", fontsize=25)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(False)


save_path_png = r"G:\CVD\5.ROC\2.3\5bio-ROC.png" 
save_path_pdf = r"G:\CVD\5.ROC\2.3\5bio-ROC.pdf"  

plt.savefig(save_path_png, dpi=300, bbox_inches='tight')  
plt.savefig(save_path_pdf, format='pdf', dpi=300, bbox_inches='tight') 

print(f"ROC : {save_path_png}  {save_path_pdf}")
plt.show()
