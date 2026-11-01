# Predictive Churn Analysis - Report

**Data:** `customer_churn_model_v2.csv` - 800 synthetic customers, 11 predictors, target `Churn` (41.9% churned). No missing values or duplicates.
**Reproduce:** `pip install -r requirements.txt && python churn_model.py` (random seed 42).

## Method
1. **Encoding:** One-Hot for Region, SubscriptionType, ContractType, PaymentMethod, AutoPay; MinMax scaling for the 6 numeric features; `CustomerID` dropped. Everything sits inside one scikit-learn `Pipeline`, so scaling is fit on training data only.
2. **Split:** 80/20 stratified (640 train / 160 test), `random_state=42`.
3. **Models:** Logistic Regression and Random Forest, each tuned with 5-fold CV (scoring ROC-AUC) on the training set only. The final model was picked by CV AUC, so the test set was never used for selection.
4. **Evaluation:** Precision, Recall, F1 and ROC-AUC on the 160 held-out customers at a 0.5 threshold.

## Results (test set, churn = positive class)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.644 | 0.586 | 0.507 | 0.544 | 0.683 |
| **Random Forest (selected)** | 0.650 | 0.578 | 0.612 | 0.594 | 0.684 |

Random Forest confusion matrix: 63 true negatives, 30 false positives, 26 false negatives, 41 true positives. Charts: `outputs/roc_curve.png`, `confusion_matrix.png`, `feature_importance.png`.

## Interpretation
- **Contract type is the strongest driver.** Month-to-Month customers churn at 59% vs 23% (One Year) and 20% (Two Year).
- **Late payments raise risk steadily:** churn is 23% with 0 late payments, 51% with 2, 68% with 4.
- Longer tenure, higher monthly charges and longer contracts lower risk in the logistic model; more support tickets raise it.

## Limitations (please read)
- **Predictive power is modest** (AUC about 0.68). The model ranks churners better than chance, but about 4 in 10 predictions at the 0.5 threshold are wrong. With only 160 test customers, these metrics could shift by several points on a different split.
- Both models score about the same on test data; Random Forest was chosen on cross-validation, and its advantage is small.
- `TotalChargesINR` is roughly tenure x monthly charges, so those three features overlap and importance is shared among them.
- The data is synthetic, so findings are for learning and do not describe real customers.
- **Risk scores:** `outputs/churn_risk_scores.csv` scores all 800 customers. Scores for rows marked `Split = train` are in-sample and optimistic; use `Split = test` rows for honest evaluation. Most scores fall in the Low/Medium bands, so rank by `ChurnRiskScore` rather than relying on the band labels.

## Suggested actions
Prioritise retention offers for Month-to-Month customers with 2+ late payments, and test incentives to move them to annual contracts.
