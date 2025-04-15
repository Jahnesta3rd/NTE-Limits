import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("CPW 2022-2025 V2.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
numeric_df = df.select_dtypes(include=[np.number])

# Best-fit distribution (simplified)
def get_best_fit(data):
    distributions = {'norm': stats.norm, 'lognorm': stats.lognorm, 'gamma': stats.gamma}
    best_p = -np.inf
    best_dist = 'norm'
    for name, dist in distributions.items():
        try:
            params = dist.fit(data)
            _, p = stats.kstest(data, name, args=params)
            if p > best_p:
                best_p = p
                best_dist = name
                best_params = params
        except:
            continue
    return best_dist, best_params

# Sidebar for category selection
category = st.sidebar.selectbox("Select a Category", numeric_df.columns)
data = df[category].dropna()
dates = df['Date']

# Fit distribution and get NTE
dist_name, params = get_best_fit(data)
dist = getattr(stats, dist_name)
nte_95 = dist.ppf(0.95, *params)
nte_99 = dist.ppf(0.99, *params)

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dates, data, label="Actual", color="blue")
ax.axhline(nte_95, color="orange", linestyle="--", label="95th Percentile NTE")
ax.axhline(nte_99, color="red", linestyle="--", label="99th Percentile NTE")
ax.fill_between(dates, nte_95, data, where=(data > nte_95), interpolate=True, color='orange', alpha=0.3)
ax.fill_between(dates, nte_99, data, where=(data > nte_99), interpolate=True, color='red', alpha=0.3)
ax.set_title(f"NTE Monitoring for {category}")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Metrics
st.metric("Max Value", f"{data.max():,.0f}")
st.metric("95th NTE", f"{nte_95:,.0f}")
st.metric("99th NTE", f"{nte_99:,.0f}")
