# 🎓 Student Dropout Rate Prediction

This project aims to build predictive models to identify students at risk of dropping out, using a real-world dataset that includes demographic, academic, and social indicators. Multiple classification techniques are applied and compared to support data-driven interventions in educational settings.

---

## 📌 Project Goals

- **Predict student dropout risk** using various machine learning models.
- **Engineer and preprocess features** from both categorical and numerical data.
- **Evaluate model performance** using accuracy, Kappa statistic, and confusion matrices.
- **Identify most important features** influencing student retention.

---

## 🧰 Tools & Techniques

- **Language**: R  
- **Libraries**: `caret`, `e1071`, `ggplot2`, `corrplot`, `pheatmap`, `klaR`  
- **Algorithms**:  
  - Logistic Regression  
  - LDA / QDA / MDA / RDA  
  - Neural Networks (NN)  
  - KNN  
  - Naive Bayes  
  - SVM  
  - PLS  
  - Penalized Models (`glmnet`)  
  - Flexible Discriminant Analysis (FDA)

---

## 📊 Workflow

1. **Data Preprocessing**  
   - Categorical encoding with `dummyVars`  
   - Removal of near-zero variance predictors  
   - Handling multicollinearity and skewed features  
   - Box-Cox transformation

2. **Model Training and Evaluation**  
   - Train/test split using `createDataPartition`  
   - 10-fold cross-validation via `trainControl`  
   - Confusion matrix evaluation on multiple models

3. **Feature Importance**  
   - Variable importance visualization using `varImp` and `ggplot2`

---


---

## 🚀 Outcome

This end-to-end analysis helps academic institutions anticipate student attrition with reasonable accuracy. The best-performing models can be used to trigger early interventions based on identified risk factors.


