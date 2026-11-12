# Datheon Machine Learning and Data Science Internship

Two supervised models, each in its own notebook. Task 1 predicts an iris species. 
Task 2 predicts a California block's median house value.

## Task 1 - Iris classification

Notebook: `task1_iris_classification.ipynb`

Data: `iris.csv` (150 flowers, 4 measurements, 3 species)

The notebook loads the data, draws a pairplot, splits 80/20, scales the measurements with `StandardScaler` fit on the training rows, and trains Logistic Regression and a Decision Tree. On the 30 test flowers both models score 0.933. The misses are between versicolor and virginica. The notebook also has confusion-matrix heatmaps and a decision boundary on the two petal measurements.

`app.py` is a small page for trying a new flower. From this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The example buttons load a setosa, a versicolor, and a virginica.

## Task 2 - Housing price regression

Notebook: `task2_housing_regression.ipynb`

Data: `housing.csv` (California block groups)

The target is `median_house_value`, in dollars. The notebook checks the columns and the 207 missing bedroom counts, plots a correlation heatmap, splits 80/20, and fills missing values from the training rows only. It adds rooms per household, bedrooms per room, people per household, and centered interactions with income. It checks multicollinearity, scales the continuous columns, and one-hot encodes `ocean_proximity`.

Ridge and Lasso are compared with ordinary least squares. On the test blocks the engineered models have an RMSE of about $71,000, an MAE of about $50,000, and an R² of 0.615. Median income is the strongest continuous weight. The notebook includes the alpha curves, the coefficients, and the residual plots.
