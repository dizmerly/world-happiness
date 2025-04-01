import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
df = pd.read_csv("world-happiness-report-2021.csv")

# Clean column names for easy access
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()


# Function to compute statistics
def compute_statistics(x, y):
    mean_x = df[x].mean()
    mean_y = df[y].mean()
    var_x = df[x].var()
    var_y = df[y].var()
    covariance = df[[x, y]].cov().iloc[0, 1]

    return mean_x, mean_y, var_x, var_y, covariance


# First plot: Ladder score vs Logged GDP per capita
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["logged_gdp_per_capita"], y=df["ladder_score"])
plt.title("Ladder Score vs Logged GDP per capita")
plt.xlabel("Logged GDP per capita")
plt.ylabel("Ladder Score")

# Linear regression for the first plot
regressor = LinearRegression().fit(df[["logged_gdp_per_capita"]], df["ladder_score"])
df["regression_line_1"] = regressor.predict(df[["logged_gdp_per_capita"]])
plt.plot(df["logged_gdp_per_capita"], df["regression_line_1"], color="blue", label="Linear Regression", linewidth=2)

# Polynomial regression (quadratic)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(df[["logged_gdp_per_capita"]])

# Generate a smooth line for the quadratic regression
X_range = np.linspace(df["logged_gdp_per_capita"].min(), df["logged_gdp_per_capita"].max(), 100).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)
poly_regressor = LinearRegression().fit(X_poly, df["ladder_score"])
y_poly_pred = poly_regressor.predict(X_range_poly)

# Plot the quadratic regression line
plt.plot(X_range, y_poly_pred, color="red", label="Quadratic Regression", linewidth=2)  # Single smooth line

# Add a legend for regression lines
plt.legend()

# Compute and display statistics for the first plot
mean_x, mean_y, var_x, var_y, covariance = compute_statistics("logged_gdp_per_capita", "ladder_score")
data_text = f"Mean of GDP: {mean_x:.4f}\nMean of Ladder Score: {mean_y:.4f}\nVariance of GDP: {var_x:.4f}\nVariance of Ladder Score: {var_y:.4f}\nCovariance: {covariance:.4f}"
plt.text(0.95, 0.05, data_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom',
         horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.show(block=True)

# Second plot: Ladder score vs Social Support
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["social_support"], y=df["ladder_score"])
plt.title("Ladder Score vs Social Support")
plt.xlabel("Social Support")
plt.ylabel("Ladder Score")

# Linear regression for the second plot
regressor = LinearRegression().fit(df[["social_support"]], df["ladder_score"])
df["regression_line_3"] = regressor.predict(df[["social_support"]])
plt.plot(df["social_support"], df["regression_line_3"], color="blue", label="Linear Regression", linewidth=2)

# Polynomial regression (quadratic)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(df[["social_support"]])

# Generate a smooth line for the quadratic regression
X_range = np.linspace(df["social_support"].min(), df["social_support"].max(), 100).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)
poly_regressor = LinearRegression().fit(X_poly, df["ladder_score"])
y_poly_pred = poly_regressor.predict(X_range_poly)

# Plot the quadratic regression line
plt.plot(X_range, y_poly_pred, color="red", label="Quadratic Regression", linewidth=2)  # Single smooth line

# Add a legend for regression lines
plt.legend()

# Compute and display statistics for the second plot
mean_x, mean_y, var_x, var_y, covariance = compute_statistics("social_support", "ladder_score")
data_text = f"Mean of Social Support: {mean_x:.4f}\nMean of Ladder Score: {mean_y:.4f}\nVariance of Social Support: {var_x:.4f}\nVariance of Ladder Score: {var_y:.4f}\nCovariance: {covariance:.4f}"
plt.text(0.95, 0.05, data_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom',
         horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.show(block=True)

# Third plot: Ladder score vs Healthy life expectancy
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["healthy_life_expectancy"], y=df["ladder_score"])
plt.title("Ladder Score vs Healthy Life Expectancy")
plt.xlabel("Healthy Life Expectancy")
plt.ylabel("Ladder Score")

# Linear regression for the third plot
regressor = LinearRegression().fit(df[["healthy_life_expectancy"]], df["ladder_score"])
df["regression_line_5"] = regressor.predict(df[["healthy_life_expectancy"]])
plt.plot(df["healthy_life_expectancy"], df["regression_line_5"], color="blue", label="Linear Regression", linewidth=2)

# Polynomial regression (quadratic)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(df[["healthy_life_expectancy"]])

# Generate a smooth line for the quadratic regression
X_range = np.linspace(df["healthy_life_expectancy"].min(), df["healthy_life_expectancy"].max(), 100).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)
poly_regressor = LinearRegression().fit(X_poly, df["ladder_score"])
y_poly_pred = poly_regressor.predict(X_range_poly)

# Plot the quadratic regression line
plt.plot(X_range, y_poly_pred, color="red", label="Quadratic Regression", linewidth=2)  # Single smooth line

# Add a legend for regression lines
plt.legend()

# Compute and display statistics for the third plot
mean_x, mean_y, var_x, var_y, covariance = compute_statistics("healthy_life_expectancy", "ladder_score")
data_text = f"Mean of Life Expectancy: {mean_x:.4f}\nMean of Ladder Score: {mean_y:.4f}\nVariance of Life Expectancy: {var_x:.4f}\nVariance of Ladder Score: {var_y:.4f}\nCovariance: {covariance:.4f}"
plt.text(0.95, 0.05, data_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom',
         horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.show(block=True)

# Fourth plot: Ladder score vs Freedom to make life choices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["freedom_to_make_life_choices"], y=df["ladder_score"])
plt.title("Ladder Score vs Freedom to Make Life Choices")
plt.xlabel("Freedom to Make Life Choices")
plt.ylabel("Ladder Score")

# Linear regression for the fourth plot
regressor = LinearRegression().fit(df[["freedom_to_make_life_choices"]], df["ladder_score"])
df["regression_line_7"] = regressor.predict(df[["freedom_to_make_life_choices"]])
plt.plot(df["freedom_to_make_life_choices"], df["regression_line_7"], color="blue", label="Linear Regression",
         linewidth=2)

# Polynomial regression (quadratic)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(df[["freedom_to_make_life_choices"]])

# Generate a smooth line for the quadratic regression
X_range = np.linspace(df["freedom_to_make_life_choices"].min(), df["freedom_to_make_life_choices"].max(), 100).reshape(
    -1, 1)
X_range_poly = poly_features.transform(X_range)
poly_regressor = LinearRegression().fit(X_poly, df["ladder_score"])
y_poly_pred = poly_regressor.predict(X_range_poly)

# Plot the quadratic regression line
plt.plot(X_range, y_poly_pred, color="red", label="Quadratic Regression", linewidth=2)  # Single smooth line

# Add a legend for regression lines
plt.legend()

# Compute and display statistics for the fourth plot
mean_x, mean_y, var_x, var_y, covariance = compute_statistics("freedom_to_make_life_choices", "ladder_score")
data_text = f"Mean of Freedom: {mean_x:.4f}\nMean of Ladder Score: {mean_y:.4f}\nVariance of Freedom: {var_x:.4f}\nVariance of Ladder Score: {var_y:.4f}\nCovariance: {covariance:.4f}"
plt.text(0.95, 0.05, data_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom',
         horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.show(block=True)
