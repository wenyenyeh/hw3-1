import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 標題
st.title("1D Logistic Regression vs SVM Comparison")

# 生成1D分類資料集
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=42)

# 訓練邏輯回歸和SVM模型
log_reg = LogisticRegression()
log_reg.fit(X, y)
svm = SVC(kernel='linear', probability=True)
svm.fit(X, y)

# 生成測試點以進行決策邊界的繪製
x_test = np.linspace(X.min() - 0.5, X.max() + 0.5, 300).reshape(-1, 1)
log_reg_probs = log_reg.predict_proba(x_test)[:, 1]
svm_probs = svm.predict_proba(x_test)[:, 1]

# 建立視覺化
fig, ax = plt.subplots()
ax.scatter(X, y, c=y, cmap='bwr', edgecolors='k', alpha=0.6, label="Data Points")
ax.plot(x_test, log_reg_probs, label="Logistic Regression", color='blue')
ax.plot(x_test, svm_probs, label="SVM (Linear Kernel)", color='green')
ax.set_xlabel("Feature")
ax.set_ylabel("Probability")
ax.set_title("Decision Boundary Comparison")
ax.legend()

# 顯示圖表
st.pyplot(fig)

# 顯示模型準確率
st.write("Logistic Regression Accuracy:", log_reg.score(X, y))
st.write("SVM Accuracy:", svm.score(X, y))
