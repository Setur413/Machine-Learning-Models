# 🚢 Titanic Survival Prediction — Machine Learning Models

> Binary classification project predicting passenger survival using the Kaggle Titanic dataset.  
> Two models compared: **Logistic Regression** vs **Random Forest Classifier**

---

## 📌 Project Overview

This project walks through a complete machine learning pipeline — from raw data exploration to model evaluation — using the classic Titanic dataset. The goal is to predict whether a passenger survived the Titanic disaster based on features like age, sex, ticket class, and fare.

Two classifiers were independently built, trained, and evaluated:

| Notebook | Model | Accuracy |
|----------|-------|----------|
| `titanic1.ipynb` | Random Forest Classifier | 74.7% |
| `titanic2.ipynb` | Logistic Regression | **78.1%** ✅ |

> **Winner:** Logistic Regression outperformed Random Forest on this dataset, likely because the survival patterns are fairly linearly separable (e.g., female + 1st class + high fare → survived).

---

## 📁 Repository Structure

```
Machine-Learning-Models/
│
├── titanic1.ipynb        # Random Forest Classifier
├── titanic2.ipynb        # Logistic Regression
└── README.md
```

---

## 🔄 ML Pipeline

Both notebooks follow the same end-to-end pipeline:

### 1. 📥 Data Loading & Exploration
- Loaded `train.csv` (891 rows × 12 columns)
- Inspected data types, shape, and null counts
- Checked for duplicates (none found)
- Identified categorical vs numerical columns

### 2. 🧹 Data Cleaning
| Column | Issue | Action |
|--------|-------|--------|
| `Cabin` | 77.1% missing | Dropped |
| `Age` | 19.87% missing | Imputed with mean |
| `Embarked` | 0.22% missing | Dropped 2 rows |
| `Name` | 891 unique values | Dropped |
| `Ticket` | 681 unique values | Dropped |

### 3. 🔢 Feature Encoding
- **Sex** → Label encoded: `male = 1`, `female = 0`
- **Embarked** → One-hot encoded via `pd.get_dummies(drop_first=True)` → `Embarked_Q`, `Embarked_S`

### 4. ⚖️ Feature Scaling
Applied `MinMaxScaler` to continuous features:
- `Age`, `Fare`, `SibSp`, `Parch` → scaled to [0, 1]

### 5. 🏋️ Model Training
- Train/Test split: **80% / 20%** (`random_state=42`)
- Trained each classifier on the same preprocessed feature set

### 6. 📊 Evaluation
- Accuracy score
- Confusion matrix
- Classification report (Precision, Recall, F1-Score)

---

## 🎯 Final Feature Set

| Feature | Description |
|---------|-------------|
| `Pclass` | Passenger class (1, 2, 3) |
| `Sex` | Gender (binary encoded) |
| `Age` | Age in years (scaled) |
| `SibSp` | Siblings/spouses aboard (scaled) |
| `Parch` | Parents/children aboard (scaled) |
| `Fare` | Ticket fare (scaled) |
| `Embarked_Q` | Departed from Queenstown |
| `Embarked_S` | Departed from Southampton |

---

## 📈 Results

### Logistic Regression (`titanic2.ipynb`)
```
Accuracy: 78.1%

Confusion Matrix:
 [[85 24]
  [15 54]]

Classification Report:
              precision  recall  f1-score  support
           0       0.85    0.78      0.81      109
           1       0.69    0.78      0.73       69
    accuracy                         0.78      178
```

### Random Forest (`titanic1.ipynb`)
```
Accuracy: 74.7%

Confusion Matrix:
 [[82 27]
  [18 51]]

Classification Report:
              precision  recall  f1-score  support
           0       0.82    0.75      0.78      109
           1       0.65    0.74      0.69       69
    accuracy                         0.75      178
```

---

## 💡 Key Findings

- 👩 **Sex** was one of the strongest predictors — female passengers had significantly higher survival rates
- 💰 **Fare** had the highest feature importance in Random Forest — a proxy for wealth and cabin location
- 🎫 **Pclass** strongly correlated with survival — 1st class passengers had far better odds
- 👶 **Age** mattered — younger passengers, especially children, were more likely to survive
- 🏠 **Embarked** had minimal predictive impact once other features were accounted for

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML models, scaling, evaluation |
| Matplotlib & Seaborn | Data visualization |
| Jupyter Notebook | Development environment |

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Setur413/Machine-Learning-Models.git
   cd Machine-Learning-Models
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. **Add the dataset**  
   Download `train.csv` from [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data) and place it in the project root.

4. **Run the notebooks**
   ```bash
   jupyter notebook
   ```
   Open `titanic1.ipynb` for Random Forest or `titanic2.ipynb` for Logistic Regression.

---

## 📌 Potential Improvements

- [ ] Extract titles from passenger names (Mr., Mrs., Miss., Master)
- [ ] Engineer `FamilySize` and `IsAlone` features
- [ ] Hyperparameter tuning with `GridSearchCV`
- [ ] Try XGBoost / LightGBM for better performance
- [ ] Use k-fold cross-validation for more robust evaluation
- [ ] Address class imbalance with SMOTE or class weighting

---

## 📄 Dataset

- **Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Size:** 891 rows × 12 columns
- **Task:** Binary classification (Survived: 0 or 1)
