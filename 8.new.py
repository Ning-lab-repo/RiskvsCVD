import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier  
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


label_mapping = {
    "heart": "Unstable angina",
    "acute": "Acute myocardial infarction",
    "chronic": "Chronic ischemic heart disease",
    "brain": "Cerebral infarction",
    "Intracerebral": "Intracerebral hemorrhage",
    "Sequelae": "Sequelae of cerebrovascular disease"
}

input_paths = [
    r"G:\CVD\3.KNN\risk+heart.xlsx",
    r"G:\CVD\3.KNN\risk+acute.xlsx",
    r"G:\CVD\3.KNN\risk+chronic.xlsx",
    r"G:\CVD\3.KNN\risk+brain.xlsx",
    r"G:\CVD\3.KNN\risk+Intracerebral.xlsx",
    r"G:\CVD\3.KNN\risk+Sequelae.xlsx",
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

    xgb_model = XGBClassifier(
        alpha=1.5,  
        reg_lambda=1.5,  
        gamma=0.2,  
        colsample_bytree=0.75,  
        subsample=0.85,  
        learning_rate=0.007, 
        max_depth=9,
        n_estimators=5200,  
        min_child_weight=3,  
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='roc_auc')


    print(f"auc：{cv_scores}")
    print(f" average-AUC ：{cv_scores.mean():.4f}")

    xgb_model.fit(X_train, y_train)

    y_pred_xgb_proba = xgb_model.predict_proba(X_test)[:, 1]  

    roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb_proba)

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba)

    label = input_path.split('/')[-1].replace('.xlsx', '').split('+')[1]
    label = label_mapping.get(label, label)  

    plt.plot(fpr_xgb, tpr_xgb, label=f"Risk Populations vs {label} (AUC = {roc_auc_xgb:.4f})")


plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate", fontsize=25)
plt.ylabel("True Positive Rate", fontsize=25)
plt.title("Risk Populations vs CVD Patients", fontsize=25)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(False)


save_path_png = r"G:\CVD\5.ROC\3.\8.Risk Populations vs CVD Patients7.png"
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')  


save_path_pdf = r"G:\CVD\5.ROC\3.\8.Risk Populations vs CVD Patients7.pdf"
plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight') 
print(f"ROCF: {save_path_pdf}")
plt.show()
