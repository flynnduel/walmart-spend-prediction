# Walmart Customer Spend Prediction

Can demographic data predict how much a customer spends in a single transaction? Partially. And the part it can't predict turned out to be as interesting as the part it could.

---

## Business context

Walmart runs hundreds of millions of transactions. At that scale, even a small lift in predicting spend by segment is worth real money on inventory, promotions, and recommendations. I treated this as a regression problem and ran two different modeling approaches against each other to see which held up.

**Target variable:** Purchase amount (continuous, dollars)  
**Dataset:** Walmart transaction records with customer demographics and product categories

---

## Key findings

Customers aged 51 to 55 spend the most per transaction, which makes them an obvious target for higher-end promotions. The bigger surprise was rural. Category C customers outspend urban customers on average, even though rural markets usually get treated as lower priority. Gender barely mattered once everything else was controlled for; men spend more in total, but that's volume, not bigger baskets. Random Forest explained about 46% of the variance in spend, and I'd rather be honest about that ceiling than dress it up.

<img width="458" height="596" alt="image" src="https://github.com/user-attachments/assets/d5296b70-8c45-4482-9019-0e7b0e670edb" />

**Figure 1:**  _Customers aged 51–55 spend the most per transaction_

<img width="278" height="610" alt="image" src="https://github.com/user-attachments/assets/437bd211-c7c1-41cd-9d87-040550ce9f78" />

**Figure 2:**  _City Category C (rural) leads in average purchase amount_

<img width="1008" height="774" alt="image" src="https://github.com/user-attachments/assets/816fa5f5-d374-4e3a-a305-6370c312b8c4" />

**Figure 3:**  _Although male customers account for a larger share of total purchases, both genders demonstrate similar average spending per transaction._

---

## Models compared

| Model | RMSE | R² | MAE |
|---|---|---|---|
| Random Forest | 3,820 | 0.46 | 2,862 |
| Elastic Net | 4,704 | 0.13 | 3,604 |

Random Forest won on every metric. Elastic Net still earned its keep by confirming which variables to ignore. It zeroed out the weak predictors automatically, and Gender was one of the first to go.

---

## What the model had to work with

The data covered age group, marital status, city category, occupation, and product category. It didn't cover income, loyalty status, promotion exposure, or how a customer spends over time. That's why R² tops out where it does. The 0.46 is a data problem, not a methods problem, and a richer dataset would move it a lot.

---

## Approach

Data prep:
- Converted the "4+" string in `Stay_In_Current_City_Years` to a number
- Kept categoricals as factors for Random Forest; dummy-encoded them for Elastic Net
- Dropped User_ID and Product_ID since identifiers carry no predictive value
- Split 70/30 with `createDataPartition()`

Modeling:
- Random Forest ran 10-fold cross-validation, tuned `mtry` across the full range, 500 trees
- Elastic Net ran 10-fold CV with lambda tuned over `10^seq(3, -3, length=100)` and alpha fixed at 1
- Both trained through `caret::train()`

Top predictors by variable importance: Product Category, then Occupation, then Age. City Category and Marital Status came next.

---

## Recommendations

Build promotions around the 51 to 55 segment, especially for bigger-ticket items. Stop writing off Category C rural markets, because the data says they overperform the assumption. Start collecting income, loyalty, and promotion-response fields, since that's the real ceiling on prediction here. For any near-term prediction work, deploy Random Forest; Elastic Net isn't accurate enough to lean on.

---

## Files

```
walmart-spend-prediction/
├── Walmart Customer Spend Prediction.R     # Cleaning, EDA, modeling, evaluation
├── Walmart Customer Spend Prediction.pdf   # Full writeup
├── walmart.csv                             # Dataset (public Kaggle Black Friday set)
└── README.md
```

---

## Tools

R, caret, randomForest, glmnet, tidyverse, ggplot2
