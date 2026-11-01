# Dataset

`dataset.csv` is the **IBM HR Analytics Employee Attrition & Performance** dataset, the same file as `WA_Fn-UseC_-HR-Employee-Attrition.csv` on [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). It is already included, so no download is needed.

- **Rows:** 1,470 employees
- **Columns:** 35, including demographics, job details, compensation, satisfaction scores and work history
- **Target:** `Attrition` (Yes/No), 16.1% Yes
- **Quality:** no missing values, no duplicate rows

`EmployeeCount`, `Over18` and `StandardHours` hold a single value, and `EmployeeNumber` is an ID. All four are dropped before modelling.

The data was created by IBM data scientists for educational use and does not describe a real company.
