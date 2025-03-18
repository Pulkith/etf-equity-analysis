# %% [markdown]
# # ETF Return & Simulation Analysis for PEAK6 Interview
# 
# This notebook analyzes historical data for the following ETFs:  
# **SPY, XLU, IGV, SMH, ARKK, XLE, QQQ, XLK**  
# We compute risk/return metrics, plot various visualizations, and perform Monte Carlo simulations to project possible price paths for 1 year (assumed as the rest of 2025).  
# 
# Our goal is to identify which ETF might have the fattest right tail – meaning the greatest chance of extreme positive returns.

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Use a clean style for plots
plt.style.use('seaborn-whitegrid')
%matplotlib inline

# %%
# Define the ETFs and download historical data
etfs = ["SPY", "XLU", "IGV", "SMH", "ARKK", "XLE", "QQQ", "XLK"]
# Download 4 years of data; adjust dates as needed
start_date = '2020-01-01'
end_date = '2023-12-31'
data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']

# Display the first few rows of data
data.head()

# %%
# Plot historical adjusted closing prices
plt.figure(figsize=(14,8))
for etf in etfs:
    plt.plot(data.index, data[etf], label=etf)
plt.title("Historical Adjusted Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# %%
# Calculate daily log returns for each ETF
log_returns = np.log(data / data.shift(1)).dropna()

# %%
# Plot histogram of log returns for each ETF
fig, axes = plt.subplots(2, 4, figsize=(20,10))
axes = axes.flatten()
for i, etf in enumerate(etfs):
    axes[i].hist(log_returns[etf], bins=50, alpha=0.7, color='steelblue')
    axes[i].set_title(f"{etf} Log Return Distribution")
    axes[i].set_xlabel("Log Return")
    axes[i].set_ylabel("Frequency")
plt.tight_layout()
plt.show()

# %%
# Calculate descriptive statistics: mean, volatility, skewness, kurtosis, and VaR (5% quantile)
desc_stats = pd.DataFrame(index=etfs, columns=["Mean", "Volatility", "Skewness", "Kurtosis", "VaR_5%"])
for etf in etfs:
    desc_stats.loc[etf, "Mean"] = log_returns[etf].mean()
    desc_stats.loc[etf, "Volatility"] = log_returns[etf].std()
    desc_stats.loc[etf, "Skewness"] = log_returns[etf].skew()
    desc_stats.loc[etf, "Kurtosis"] = log_returns[etf].kurtosis()
    desc_stats.loc[etf, "VaR_5%"] = np.percentile(log_returns[etf], 5)

desc_stats = desc_stats.astype(float)
print("Descriptive Statistics:")
print(desc_stats)

# %%
# Bar charts for skewness, kurtosis, and volatility for a side-by-side comparison
fig, axs = plt.subplots(1, 3, figsize=(18,6))
desc_stats['Skewness'].plot(kind='bar', ax=axs[0], color='skyblue')
axs[0].set_title("Skewness")
axs[0].set_ylabel("Skewness Value")
desc_stats['Kurtosis'].plot(kind='bar', ax=axs[1], color='lightgreen')
axs[1].set_title("Kurtosis")
desc_stats['Volatility'].plot(kind='bar', ax=axs[2], color='salmon')
axs[2].set_title("Volatility")
plt.tight_layout()
plt.show()

# %%
# Plot the correlation heatmap of log returns to see how ETFs move relative to each other
plt.figure(figsize=(10,8))
corr = log_returns.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix of Log Returns")
plt.show()

# %%
# ### Monte Carlo Simulation for 1-Year Forecast (252 Trading Days)
# 
# For each ETF, we simulate future price paths using the Geometric Brownian Motion (GBM) model:
#
# \[
# S_{t+1} = S_t \times \exp\left((\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} \, \epsilon\right)
# \]
#
# where \(\epsilon\) is drawn from a standard normal distribution.
#
# We annualize the mean and volatility from our daily log returns for simulation purposes.

T = 1.0     # Time horizon of 1 year
N = 252     # Trading days in a year
dt = T / N  # Time step
num_simulations = 1000  # Number of simulation paths

simulation_results = {}

for etf in etfs:
    S0 = data[etf].iloc[-1]  # current price (last observed)
    # Annualize mean and volatility from daily returns
    mu = log_returns[etf].mean() * 252  
    sigma = log_returns[etf].std() * np.sqrt(252)
    
    sim_prices = np.zeros((N+1, num_simulations))
    sim_prices[0] = S0
    
    # Generate simulated price paths
    for t in range(1, N+1):
        z = np.random.normal(size=num_simulations)
        sim_prices[t] = sim_prices[t-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*z)
    
    simulation_results[etf] = sim_prices
    
    # Plot sample simulation paths (first 10 paths)
    plt.figure(figsize=(10,6))
    plt.plot(sim_prices[:, :10], lw=0.8)
    plt.title(f"Monte Carlo Simulation Paths for {etf}")
    plt.xlabel("Trading Days")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()
    
    # Plot histogram of final simulated prices
    plt.figure(figsize=(10,6))
    plt.hist(sim_prices[-1], bins=50, alpha=0.7, color='mediumseagreen')
    plt.title(f"Distribution of Final Prices for {etf} after 1 Year")
    plt.xlabel("Final Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# %%
# Calculate simulation statistics for each ETF: mean final price, median, skewness, kurtosis
simulation_stats = pd.DataFrame(index=etfs, columns=["Sim_Mean_Final_Price", "Sim_Median_Final_Price", "Sim_Skewness", "Sim_Kurtosis"])
for etf in etfs:
    final_prices = simulation_results[etf][-1]
    simulation_stats.loc[etf, "Sim_Mean_Final_Price"] = final_prices.mean()
    simulation_stats.loc[etf, "Sim_Median_Final_Price"] = np.median(final_prices)
    simulation_stats.loc[etf, "Sim_Skewness"] = stats.skew(final_prices)
    simulation_stats.loc[etf, "Sim_Kurtosis"] = stats.kurtosis(final_prices)
    
simulation_stats = simulation_stats.astype(float)
print("Monte Carlo Simulation Descriptive Statistics:")
print(simulation_stats)

# %%
# QQ Plots to check the normality of the historical log returns for each ETF
fig, axes = plt.subplots(2, 4, figsize=(20,12))
axes = axes.flatten()
for i, etf in enumerate(etfs):
    sm.qqplot(log_returns[etf], line='s', ax=axes[i])
    axes[i].set_title(f"QQ Plot for {etf}")
plt.tight_layout()
plt.show()
