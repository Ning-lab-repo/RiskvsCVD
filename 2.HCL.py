import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from xgboost import XGBClassifier  
label_mapping = {
    "heart": "Unstable angina",
    "acute": "Acute myocardial infarction",
    "chronic": "Chronic ischemic heart disease",
    "brain": "Cerebral infarction",
    "Intracerebral": "Intracerebral hemorrhage",
    "Sequelae": "Sequelae of cerebrovascular disease",
    "CVD": "CVD patients"
}

input_paths = [
    r"G:\CVD\3.KNN\HCL\HCL1+heart.xlsx",
    r"G:\CVD\3.KNN\HCL\HCL2+acute.xlsx",
    r"G:\CVD\3.KNN\HCL\HCL3+chronic.xlsx",
    r"G:\CVD\3.KNN\HCL\HCL4+brain.xlsx",
    r"G:\CVD\3.KNN\HCL\HCL5+Intracerebral.xlsx",
    r"G:\CVD\3.KNN\HCL\HCL6+Sequelae.xlsx",
    r"G:\CVD\3.KNN\HCL\2HCL7+CVD.xlsx"
]


plt.figure(figsize=(10, 8))


for input_path in input_paths:
    data = pd.read_excel(input_path)

    
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    
    scaler = StandardScaler()
    X = data.iloc[:, 2:59]  
    X_standardized = scaler.fit_transform(X)  

    
    y = data.iloc[:, -1]    
    y = y.apply(lambda x: 0 if x in [1, 2, 3] else 1)

    
    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, 
                                                        random_state=42, stratify=y)

    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    
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

 
    xgb = XGBClassifier(
    **xgb_params, 
    random_state=42, 
    use_label_encoder=False,  
    eval_metric='auc', 
    scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)  
)
  
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
    cv_scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='roc_auc')


    print(f"AUC ：{cv_scores}")
    print(f" AUC ：{cv_scores.mean():.4f}")

 
    xgb.fit(X_train, y_train)

    y_pred_rf_proba = xgb.predict_proba(X_test)[:, 1] 

    roc_auc_rf = roc_auc_score(y_test, y_pred_rf_proba)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_proba)

    label = input_path.split('/')[-1].replace('.xlsx', '').split('+')[1]
    label = label_mapping.get(label, label) 

    plt.plot(fpr_rf, tpr_rf, label=f"HCL vs {label} (AUC = {roc_auc_rf:.4f})")

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate", fontsize=25)
plt.ylabel("True Positive Rate", fontsize=25)
plt.title("Hyperlipidemia", fontsize=25)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(False)


save_path = r"G:\CVD\5.ROC\7.HCL\HCL.png"

save_path_png = r"G:\CVD\5.ROC\7.HCL\2.HCL.png"
plt.savefig(save_path_png, dpi=300, bbox_inches='tight') 
print(f"ROC : {save_path_png}")

save_path_pdf = r"G:\CVD\5.ROC\7.HCL\2.HCL.pdf"
plt.savefig(save_path_pdf, bbox_inches='tight') 
print(f"ROC : {save_path_pdf}")

plt.show()
