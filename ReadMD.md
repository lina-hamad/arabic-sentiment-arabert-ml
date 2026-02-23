# Arabic Sentiment Analysis (Fast Version) ⚡

This project performs **Arabic sentiment classification** on tweets using a hybrid feature representation:
- **AraBERT contextual embeddings (768-dim)**
- **12 handcrafted sentiment features**
- Models: **Decision Tree**, **Random Forest**, **Naive Bayes**, and an optional **Neural Network**
- Handles class imbalance using **SMOTE**
- Includes evaluation metrics + confusion matrices + comparison plots

---

## 1) Dataset

- Input file: `dataset.txt` (UTF-8)
- Format: **Tab-separated** (tweet \t label)
- Labels: `POS`, `NEG`, `OBJ`  
  - `NEUTRAL` was mapped to `OBJ` for consistency.

### Class Distribution (After Loading)
Total samples loaded: **10,006**
- **OBJ:** 7,523  
- **NEG:** 1,684  
- **POS:** 799  

This indicates a strong **class imbalance**, where `OBJ` dominates the dataset.

---

## 2) Preprocessing

The following cleaning pipeline was applied to each tweet:
- Remove URLs (`http...`, `www...`)
- Remove mentions (`@username`)
- Remove hashtag symbol but keep the word (`#topic` → `topic`)
- Remove Arabic diacritics
- Normalize Arabic characters:
  - (إ/أ/آ/ا → ا), (ى → ي), (ؤ/ئ → ء), (ة → ه)
- Reduce character repetition (elongation)
- Remove English letters and numbers
- Remove punctuation and extra whitespace

After preprocessing:
- Remaining samples: **10,006** (no tweets removed in this run)

---

## 3) Feature Extraction

### A) Handcrafted Features (12)
Extracted sentiment-indicative features including:
- Positive/Negative emoji counts
- Positive/Negative word counts (lexicon-based)
- Negation count (e.g., ما, لا, لم, لن...)
- Exclamation/question marks count
- Character count, word count
- Repeated characters count
- Average word length
- Sentiment score = (positive signals) - (negative signals)

✅ Total handcrafted features: **12**

### B) AraBERT Embeddings (Transformer Features)
- Model: `aubmindlab/bert-base-arabertv2`
- Uses Transformer self-attention to capture contextual meaning.
- Tweet embedding: **[CLS] token vector**
- Output shape: **(10006, 768)**

✅ Total embedding features: **768**

### Final Feature Vector
Total features per tweet:
- **780 features** = 12 handcrafted + 768 AraBERT

---

## 4) Train/Validation/Test Split

Stratified split to preserve class distribution:
- **Train:** 60% (6,003)
- **Validation:** 20% (2,001)
- **Test:** 20% (2,002)

---

## 5) Handling Class Imbalance (SMOTE)

SMOTE was applied to the **training set only**:
- After SMOTE: **13,539** samples
- Balanced distribution:
  - OBJ: 4,513
  - NEG: 4,513
  - POS: 4,513

---

## 6) Scaling

All features were scaled using `StandardScaler`:
- Fit on training set (after SMOTE)
- Applied to validation and test sets

Saved scaler:
- `scaler.pkl`

---

## 7) Models & Hyperparameter Tuning

GridSearchCV settings:
- **cv=3**
- scoring = **weighted F1-score** (`f1_weighted`)
- Reduced grid for faster training.

### A) Decision Tree
Best params:
- `max_depth=25`
- `min_samples_leaf=2`
- `min_samples_split=5`

Validation Accuracy: **0.5642**  
Test Accuracy: **0.5619**  
Weighted F1: **0.5949**

Saved: `decision_tree_model.pkl`

---

### B) Random Forest ✅ (Best Model)
Best params:
- `n_estimators=300`
- `max_depth=25`
- `min_samples_split=5`

Validation Accuracy: **0.7286**  
Test Accuracy: **0.7517**  
Weighted F1: **0.7380**

Saved: `random_forest_model.pkl`

---

### C) Naive Bayes (GaussianNB)
Best params:
- `var_smoothing=0.001`

Validation Accuracy: **0.5582**  
Test Accuracy: **0.5849**  
Weighted F1: **0.6201**

Saved: `naive_bayes_model.pkl`

---

### D) Neural Network (Optional)
Architecture:
- Dense(128, ReLU) + Dropout(0.3)
- Dense(64, ReLU) + Dropout(0.2)
- Dense(softmax)

Training:
- EarlyStopping on validation loss
- Converged after **7 epochs**

Test Accuracy: **0.6808**  
Weighted F1: **0.7006**

Saved:
- `neural_network_model.h5`
- `label_encoder_nn.pkl`

---

## 8) Evaluation Results

### Test Set Classification Summary
Model comparison (Accuracy / Weighted F1):

| Model | Test Accuracy | Weighted F1 |
|------|--------------:|------------:|
| Decision Tree | 0.5619 | 0.5949 |
| Random Forest | **0.7517** | **0.7380** |
| Naive Bayes | 0.5849 | 0.6201 |
| Neural Network | 0.6808 | 0.7006 |

🏆 **Best Model: Random Forest**  
- Accuracy: **0.7517**
- Weighted F1: **0.7380**

---

## 9) Visualizations

Generated plots:
- `confusion_matrices.png` (confusion matrices for all models)
- `performance_comparison.png` (Accuracy + F1 comparison)

---

## 10) Output Files

The pipeline saves:
- `final_results.csv`
- `confusion_matrices.png`
- `performance_comparison.png`
- `decision_tree_model.pkl`
- `random_forest_model.pkl`
- `naive_bayes_model.pkl`
- `scaler.pkl`
- (optional) `neural_network_model.h5`
- (optional) `label_encoder_nn.pkl`

---

## 11) Requirements

Install dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn torch transformers emoji tqdm matplotlib seaborn tensorflow
