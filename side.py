import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Load the dataset
df = pd.read_csv("world-happiness-report-2021.csv")

# Clean column names for easy access
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()


# Function to compute statistics for the data
def compute_statistics(x, y):
    mean_x = df[x].mean()
    mean_y = df[y].mean()
    var_x = df[x].var()
    var_y = df[y].var()
    covariance = df[[x, y]].cov().iloc[0, 1]

    # Compute correlation coefficient (R) and R^2
    correlation_matrix = df[[x, y]].corr()
    r = correlation_matrix.iloc[0, 1]  # Pearson correlation coefficient
    r_squared = r ** 2  # Coefficient of determination

    return mean_x, mean_y, var_x, var_y, covariance, r, r_squared


# Function to generate scatter plots with regression lines
def plot_regression(x, y, title, xlabel):
    plt.figure(figsize=(8, 6))

    # Scatter plot
    sns.scatterplot(x=df[x], y=df[y])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Happiness Score")

    # Compute statistics
    mean_x, mean_y, var_x, var_y, covariance, r, r_squared = compute_statistics(x, y)

    # Linear regression
    linear_model = LinearRegression()
    linear_model.fit(df[[x]], df[y])
    df["linear_pred"] = linear_model.predict(df[[x]])
    plt.plot(df[x], df["linear_pred"], color="blue", label="Linear Regression", linewidth=2)

    # Polynomial regression (quadratic)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(df[[x]])

    X_range = np.linspace(df[x].min(), df[x].max(), 100).reshape(-1, 1)
    X_range_poly = poly_features.transform(X_range)

    poly_model = LinearRegression()
    poly_model.fit(X_poly, df[y])
    y_poly_pred = poly_model.predict(X_range_poly)

    plt.plot(X_range, y_poly_pred, color="red", label="Quadratic Regression", linewidth=2)  # Smooth quadratic line

    # Add legend
    plt.legend()

    # Display statistics in the plot
    data_text = (
        f"Mean {xlabel}: {mean_x:.2f}\n"
        f"Mean Happiness: {mean_y:.2f}\n"
        f"Var {xlabel}: {var_x:.2f}\n"
        f"Var Happiness: {var_y:.2f}\n"
        f"Covariance: {covariance:.2f}\n"
        f"R: {r:.2f}\n"
        f"R²: {r_squared:.2f}"
    )

    plt.text(0.95, 0.05, data_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.7))


# Function to generate 3D plots
def plot_3d(x, y, z, title, xlabel, ylabel, zlabel):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot in 3D
    ax.scatter(df[x], df[y], df[z], color='b', label='Data Points')

    # Compute regression line in 3D
    # Linear regression for 3D
    X = df[[x, y]]
    Y = df[z]
    linear_model = LinearRegression()
    linear_model.fit(X, Y)

    # Create a meshgrid for the plane
    x_range = np.linspace(df[x].min(), df[x].max(), 100)
    y_range = np.linspace(df[y].min(), df[y].max(), 100)
    x_range, y_range = np.meshgrid(x_range, y_range)
    z_range = linear_model.predict(np.c_[x_range.ravel(), y_range.ravel()])
    z_range = z_range.reshape(x_range.shape)

    # Plot the plane
    ax.plot_surface(x_range, y_range, z_range, color='r', alpha=0.3)

    # Label the axes and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Show the plot
    plt.show()


# Run the plots
plot_regression("logged_gdp_per_capita", "ladder_score", "Happiness vs GDP per Capita", "GDP per Capita (log scale)")
plot_regression("social_support", "ladder_score", "Happiness vs Social Support", "Social Support")
plot_regression("healthy_life_expectancy", "ladder_score", "Happiness vs Life Expectancy", "Healthy Life Expectancy")
plot_regression("freedom_to_make_life_choices", "ladder_score", "Happiness vs Freedom", "Freedom to Make Life Choices")

# Now for 3D plots
plot_3d("logged_gdp_per_capita", "social_support", "ladder_score", "Happiness vs GDP vs Social Support",
        "GDP per Capita", "Social Support", "Happiness")
plot_3d("healthy_life_expectancy", "freedom_to_make_life_choices", "ladder_score",
        "Happiness vs Life Expectancy vs Freedom", "Life Expectancy", "Freedom to Make Life Choices", "Happiness")

