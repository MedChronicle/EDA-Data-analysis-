import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

n_months = 144
dates = pd.date_range(start='1949-01-01', periods=n_months, freq='MS')

trend = np.linspace(100, 500, n_months)
seasonal_yearly = 50 * np.sin(np.linspace(0, 12 * np.pi, n_months))
seasonal_growth = np.linspace(1.0, 1.8, n_months)
noise = np.random.normal(loc=0.0, scale=15.0, size=n_months)
passengers = (trend + seasonal_yearly * seasonal_growth + noise).astype(int)
passengers = np.clip(passengers, a_min=50, a_max=None)

df = pd.DataFrame({
    'Date': dates,
    'Passengers': passengers
})

df.to_csv('/mnt/user-data/outputs/airline-passengers.csv', index=False)

df = pd.read_csv('/mnt/user-data/outputs/airline-passengers.csv')

print(df.shape)

print(df.head())

df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)

df = df.set_index('Date')
print(df.tail(3))

print(df.sample(5, random_state=0))

print(df.loc['1955-06'])

print(df.loc['1955'])

print(df.loc['1955-01':'1955-12'])

sns.set_style('darkgrid')
sns.set(rc={'figure.figsize': (11, 4)})
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 150

fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=df, x=df.index, y='Passengers', ax=ax, color='steelblue', linewidth=1.5)
ax.set_title('Monthly Airline Passengers Over Time (1949-1960)')
ax.set_ylabel('Number of Passengers')
ax.set_xlabel('Date')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/passengers_timeseries.png')
plt.show()

df['Rolling_Mean_12M'] = df['Passengers'].rolling(window=12).mean()

fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=df, x=df.index, y='Passengers', ax=ax, label='Monthly Passengers', alpha=0.5)
sns.lineplot(data=df, x=df.index, y='Rolling_Mean_12M', ax=ax, label='12-Month Rolling Mean', color='crimson', linewidth=2)
ax.set_title('Airline Passengers with 12-Month Rolling Mean (Trend)')
ax.set_ylabel('Number of Passengers')
ax.set_xlabel('Date')
ax.legend()
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/passengers_trend.png')
plt.show()

yearly_avg = df['Passengers'].resample('YE').mean()

fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(x=yearly_avg.index.year, y=yearly_avg.values, ax=ax, color='mediumseagreen')
ax.set_title('Average Yearly Passenger Count')
ax.set_ylabel('Average Passengers')
ax.set_xlabel('Year')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/passengers_yearly_avg.png')
plt.show()

df['Cumulative_Passengers'] = df['Passengers'].cumsum()
print(df.head())

fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=df, x=df.index, y='Cumulative_Passengers', ax=ax, color='darkorange', linewidth=2)
ax.set_title('Cumulative Sum of Airline Passengers Over Time')
ax.set_ylabel('Cumulative Passengers')
ax.set_xlabel('Date')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/passengers_cumulative.png')
plt.show()

df.to_csv('/mnt/user-data/outputs/airline-passengers-analyzed.csv')
