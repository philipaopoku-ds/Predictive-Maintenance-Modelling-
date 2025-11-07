# 🧠 Predictive Maintenance System

## 🔍 Overview
**PrediTech** is a Predictive Maintenance System designed to detect potential machine failures using real-time sensor data and machine learning models.  
The project covers the complete machine learning pipeline — from data preprocessing to deployment — enabling industries to **predict failures before they occur**, reducing downtime and operational costs.

---

## 🚀 Key Steps
1. **Data Import and Cleaning** – Handle missing values and inconsistent data.  
2. **Feature Engineering** – Derived new metrics such as:
   - `Power`  
   - `Temp_Diff` (Temperature Difference)  
   - `Torque_Speed_Ratio`  
3. **Exploratory Data Analysis (EDA)** – Identify patterns and correlations.  
4. **Outlier and Correlation Analysis** – Evaluate feature interactions and retain meaningful outliers.  
5. **Feature Scaling and Encoding** – Normalize and prepare data for modeling.  
6. **Model Comparison** – Evaluated multiple algorithms:
   - Logistic Regression  
   - Random Forest  
   - Support Vector Machine (SVM)  
   - XGBoost  
   - CatBoost  
   - LightGBM  
7. **Model Optimization** – Tuned parameters using `RandomizedSearchCV`.  
8. **Deployment** – Built a **Streamlit app** for real-time failure detection.

---

## 🛠️ Tools & Technologies
- **Languages & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **ML Frameworks:** LightGBM, XGBoost, CatBoost  
- **Deployment:** Streamlit  
- **Serialization:** Pickle  
- **Development Environment:** Jupyter Notebook, VS Code  

---

## 📊 Model Performance

| Metric | Score |
|:-------|:------:|
| **Accuracy** | 99% |
| **Precision** | 98% |
| **Recall** | 81% |
| **F1-Score** | 0.89 |

**Best Model:** LightGBM  
After optimization, the LightGBM model achieved exceptional predictive strength and stability.

---

## 💡 Key Insights
- **Tool Wear**, **Power**, and **Rotational Speed** are the most critical features influencing machine failure.  
- **Temperature Difference** helps detect stress and overheating conditions.  
- Outliers were **retained** as they often indicate genuine stress signals rather than noise.  
- The system enables **proactive maintenance**, minimizing costly downtime.

---

## 🌐 Streamlit App
**Live App:** [PrediTech – Predictive Maintenance System](https://preditech-predictive-maintenance.streamlit.app/)

### App Features
- Real-time failure prediction using sensor input fields  
- Automatic scaling and encoding for consistent input handling  
- Visual and textual output:
  - ✅ *Normal Operation*  
  - ⚠️ *Failure Detected*

---
### 🏁 Conclusion
This project successfully developed a Predictive Maintenance System capable of identifying potential machine failures before they occur.
The optimized LightGBM model achieved outstanding accuracy and reliability, enabling early detection of failures, minimizing downtime, and cutting maintenance costs.

By integrating machine learning with real-time monitoring, PrediTech empowers industries to transition from reactive to proactive maintenance, improving efficiency, safety, and productivity.
The Streamlit deployment ensures the system is easily accessible to field engineers and technicians for real-world use.

---


## 🧩 How to Use

### Clone the Repository
```bash
git clone https://github.com/benmill52/predictive-maintenance-system.git

