## Preprocessing Steps
All preprocessing is implemented in the main notebook and includes:

**Missing Value Handling:**
- Age: 177 missing values (19.9%) filled with median (28.0 years)
- Cabin: 687 missing values (77.2%) - extracted deck letter for CabinDeck feature, then dropped
- Embarked: 2 missing values (0.2%) filled with mode (S - Southampton)
- Fare: No missing values

**Feature Engineering (6 New Features):**
1. Title: Extracted from passenger names (Mr=1, Miss=2, Mrs=3, Master=4, Rare=5)
2. FamilySize: SibSp + Parch + 1
3. IsAlone: Binary indicator (1 if traveling alone, 0 otherwise)
4. Age*Class: Interaction feature (Age × Pclass)
5. CabinDeck: First letter of cabin number, encoded to numeric (0-8)
6. HasCabin: Binary indicator (1 if cabin exists, 0 if missing)

**Categorical Encoding:**
- Sex: Encoded to numeric (0=female, 1=male)
- Embarked: Encoded to numeric (C=0, Q=1, S=2)
- CabinDeck: Encoded to numeric (0-8)

**Columns Dropped:**
PassengerId, Name, Ticket, Cabin, SibSp, Parch

**Final Dataset:**
- Training: X_train (891, 11), Y_train (891,)
- Testing: X_test (418, 11)
- Final features: Pclass, Sex, Age, Fare, Embarked, Title, FamilySize, IsAlone, Age*Class, CabinDeck, HasCabin

## Models Trained

### Decision Tree Classifier
- **Parameters:** max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42
- **Training Accuracy:** 85.07%
- **5-Fold Cross-Validation:** 81.03% average (Std Dev: 2.23%)
- **Fold Scores:** [81.56%, 82.02%, 80.90%, 76.97%, 83.71%]

### Random Forest Classifier
- **Parameters:** n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42
- **Training Accuracy:** 85.30%
- **5-Fold Cross-Validation:** 83.05% average (Std Dev: 1.63%)
- **Fold Scores:** [82.12%, 82.02%, 85.96%, 81.46%, 83.71%]

## Results & Comparison
- **Random Forest is superior** with 83.05% accuracy vs 81.03% for Decision Tree
- **Performance improvement:** 2.02 percentage points (2.49% relative improvement)
- **Stability:** RF has lower variance (1.63% vs 2.23%) - more consistent across folds
- **Generalization:** RF training-CV gap (2.25%) is smaller than DT (4.04%), indicating better generalization

## Key Findings
1. Random Forest ensemble reduces variance through averaging
2. Feature randomness creates diverse, decorrelated trees
3. All 6 engineered features contributed to model performance
4. Title and Sex are the most important features (27.87% and 25.01% respectively)

## Files Included
- `titanic-data-science-solutions-for-task1.ipynb`: Main notebook with preprocessing, model training, and analysis
- `decision_tree.pdf`: Visualization of the Decision Tree structure


## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, matplotlib, graphviz
