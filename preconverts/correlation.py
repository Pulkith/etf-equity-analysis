# %% [markdown]
# ## 8. Correlation Analysis Across ETFs
# 
# In this section we calculate the pairwise correlation matrix for the historical daily log returns of the ETFs.
# We then visualize the full correlation matrix as a heatmap.
# Next, we extract the upper-triangular (pairwise) correlations, plot their histogram, and finally bin these correlations into low, medium, and high categories and display a count plot.

# %%
# Compute daily log returns if not already computed
log_returns = np.log(data / data.shift(1)).dropna()

# Compute the full correlation matrix
corr_matrix = log_returns.corr()

# Plot the correlation matrix as a heatmap
import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix of ETF Log Returns")
plt.xlabel("ETF")
plt.ylabel("ETF")
plt.show()

# %%
# Extract pairwise correlations (upper triangular only)
# Create a mask for the upper triangle (excluding the diagonal)
mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
pairwise_corr = corr_matrix.where(mask).stack()  # stack removes NaNs
pairwise_corr_df = pairwise_corr.reset_index()
pairwise_corr_df.columns = ['ETF1', 'ETF2', 'Correlation']

print("Pairwise Correlations:")
print(pairwise_corr_df)

# %%
# Plot a histogram of the pairwise correlation coefficients
plt.figure(figsize=(8,6))
sns.histplot(pairwise_corr_df['Correlation'], bins=10, kde=True, color='mediumseagreen')
plt.title("Histogram of Pairwise ETF Correlations")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Count")
plt.show()

# %%
# Bin the correlation coefficients into three categories:
# Low: < 0.3, Medium: 0.3 to 0.7, High: > 0.7
bins = [-1, 0.3, 0.7, 1]
labels = ['Low (<0.3)', 'Medium (0.3-0.7)', 'High (>0.7)']
pairwise_corr_df['Corr_Bin'] = pd.cut(pairwise_corr_df['Correlation'], bins=bins, labels=labels)

# Display the count of each bin
bin_counts = pairwise_corr_df['Corr_Bin'].value_counts().sort_index()
print("\nBinned Correlation Counts:")
print(bin_counts)

# Plot a count plot for the binned correlations
plt.figure(figsize=(8,6))
sns.countplot(x='Corr_Bin', data=pairwise_corr_df, order=labels, palette='viridis')
plt.title("Count of ETF Pairs by Correlation Category")
plt.xlabel("Correlation Category")
plt.ylabel("Number of ETF Pairs")
plt.show()
