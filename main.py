import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv("world-happiness-report-2021.csv")

# Clean column names for easy access
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# Rename columns for better readability
df.rename(columns={
    "ladder_score": "happiness_score",
    "logged_gdp_per_capita": "gdp_per_capita_log",
    "social_support": "social_support_index",
    "healthy_life_expectancy": "life_expectancy_years",
    "freedom_to_make_life_choices": "freedom_of_choice_score"
}, inplace=True)


# Function to compute statistics
def compute_statistics(x, y):
    mean_x = df[x].mean()
    mean_y = df[y].mean()
    var_x = df[x].var()
    var_y = df[y].var()
    covariance = df[[x, y]].cov().iloc[0, 1]
    return mean_x, mean_y, var_x, var_y, covariance


# Function to plot scatter plot with regression lines
def plot_with_regression(ax, x, y, title, xlabel, ylabel):
    mean_x, mean_y, var_x, var_y, covariance = compute_statistics(x, y)

    # Scatter plot
    sns.scatterplot(x=df[x], y=df[y], label="Data Points", color="blue", ax=ax)

    # Generate points for regression
    x_vals = np.linspace(df[x].min(), df[x].max(), 100)

    # Linear Regression
    linear_coeffs = np.polyfit(df[x], df[y], 1)
    linear_fit = np.polyval(linear_coeffs, x_vals)
    ax.plot(x_vals, linear_fit, label="Linear Fit (Red)", color="red")

    # Quadratic Regression
    quadratic_coeffs = np.polyfit(df[x], df[y], 2)
    quadratic_fit = np.polyval(quadratic_coeffs, x_vals)
    ax.plot(x_vals, quadratic_fit, label="Quadratic Fit (Green)", color="green")

    # Add statistics as a small box at the bottom right
    stats_text = (f"Mean {xlabel}: {mean_x:.2f}\nMean {ylabel}: {mean_y:.2f}\n"
                  f"Var {xlabel}: {var_x:.2f}\nVar {ylabel}: {var_y:.2f}\n"
                  f"Cov: {covariance:.2f}")

    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'),
            verticalalignment='bottom', horizontalalignment='right')

    # Set plot labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Show legend with regression line colors
    ax.legend(loc="upper left")


# Create a figure with multiple subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Generate plots in a grid layout with improved labels
plot_with_regression(axes[0, 0], "gdp_per_capita_log", "happiness_score",
                     "Happiness vs GDP per Capita",
                     "GDP per Capita (Log Scale)", "Happiness Score")

plot_with_regression(axes[0, 1], "social_support_index", "happiness_score",
                     "Happiness vs Social Support",
                     "Social Support Index", "Happiness Score")

plot_with_regression(axes[1, 0], "life_expectancy_years", "happiness_score",
                     "Happiness vs Life Expectancy",
                     "Life Expectancy (Years)", "Happiness Score")

plot_with_regression(axes[1, 1], "freedom_of_choice_score", "happiness_score",
                     "Happiness vs Freedom of Choice",
                     "Freedom of Choice Score", "Happiness Score")

# Adjust layout for clarity
plt.tight_layout()

# Show all plots at the same time
plt.show()
