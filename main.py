import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("world-happiness-report-2021.csv")

# Clean column names for easy access
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# Turn on interactive mode so plots stay open
plt.ion()

# First plot: Ladder score vs Logged GDP per capita
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["logged_gdp_per_capita"], y=df["ladder_score"])
plt.title("Ladder Score vs Logged GDP per capita")
plt.xlabel("Logged GDP per capita")
plt.ylabel("Ladder Score")

# Second plot: Ladder score vs Social Support
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["social_support"], y=df["ladder_score"])
plt.title("Ladder Score vs Social Support")
plt.xlabel("Social Support")
plt.ylabel("Ladder Score")

# Third plot: Ladder score vs Healthy life expectancy
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["healthy_life_expectancy"], y=df["ladder_score"])
plt.title("Ladder Score vs Healthy Life Expectancy")
plt.xlabel("Healthy Life Expectancy")
plt.ylabel("Ladder Score")

# Fourth plot: Ladder score vs Freedom to make life choices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["freedom_to_make_life_choices"], y=df["ladder_score"])
plt.title("Ladder Score vs Freedom to Make Life Choices")
plt.xlabel("Freedom to Make Life Choices")
plt.ylabel("Ladder Score")
plt.show(block=True)