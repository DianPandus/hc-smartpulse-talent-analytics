# Dataset Setup Instructions

## IBM HR Analytics Attrition Dataset

This project requires the IBM HR Analytics Employee Attrition dataset.

### Download Options:

#### Option 1: Kaggle (Recommended)
1. Visit: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
2. Click "Download" button
3. Extract the ZIP file
4. Copy `WA_Fn-UseC_-HR-Employee-Attrition.csv` to the `data/` folder in this project

#### Option 2: Alternative Sources
- IBM Watson Analytics Community
- GitHub repositories with sample datasets

### File Location

Place the CSV file here:
```
HC-SmartPulse/
└── data/
    └── WA_Fn-UseC_-HR-Employee-Attrition.csv
```

### Dataset Characteristics

- **Rows**: ~1,470 employee records
- **Columns**: 35 features including:
  - Demographics (Age, Gender, Marital Status)
  - Job information (Department, Role, Level, Income)
  - Satisfaction metrics (Job, Environment, Relationship)
  - Work history (Years at company, promotions, training)
  - Target variable: **Attrition** (Yes/No)

### Data Quality

- No missing values
- Mix of categorical and numerical features
- Imbalanced target (84% No, 16% Yes)

### Verification

After placing the file, verify it's correctly located:

**Windows PowerShell:**
```powershell
Test-Path data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

**Expected output**: `True`

### Next Steps

Once the dataset is in place, proceed with:

1. Data processing: `python src/data_processing.py`
2. Model training: `python src/model_training.py`
3. Dashboard launch: `streamlit run app.py`

---

**Note**: This dataset is publicly available and commonly used for educational and research purposes. Make sure to comply with any applicable terms of use.
