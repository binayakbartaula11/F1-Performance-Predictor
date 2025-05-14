import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Make sure output directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Set consistent plot styling for clean visualization
plt.style.use('fivethirtyeight')
sns.set_palette("bright")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Loading F1 performance data...")
# Load the F1 performance dataset (2018-2024 seasons)
df = pd.read_csv('f1_performance_2018_2024.csv')

# Rename columns to follow Python naming conventions
df = df.rename(columns={
    'Driver Experience': 'Driver_Experience',
    'Average Qualifying Position': 'Avg_Qualifying_Position',
    'Average Race Position': 'Avg_Race_Position',
    'Pit Stops': 'Pit_Stops',
    'Fastest Lap Times': 'Fastest_Lap_Times',
    'Driver Points': 'Driver_Points'
})

print("Building prediction model...")
# Prepare features (X) and target variable (y)
X = df[['Driver_Experience', 'Avg_Qualifying_Position', 'Avg_Race_Position',
        'Pit_Stops', 'Fastest_Lap_Times']]
y = df['Driver_Points']

# Build and train the linear regression model using all data
model = LinearRegression()
model.fit(X, y)

print("Generating 2025 season predictions...")
# Extract 2024 data as baseline for 2025 predictions
drivers_2024 = df[df['Year'] == 2024].copy()

# Project 2025 data by incrementing driver experience
drivers_2025 = drivers_2024.copy()
drivers_2025['Year'] = 2025
drivers_2025['Driver_Experience'] += 1

# Add Lando Norris to the 2025 season projections
lando_norris = pd.DataFrame({
    'Year': [2025],
    'Driver': ['Lando Norris'],
    'Driver_Experience': [7],
    'Avg_Qualifying_Position': [3.5],
    'Avg_Race_Position': [3.0],
    'Pit_Stops': [2.0],
    'Fastest_Lap_Times': [82.0],
    'Driver_Points': [None]
})

# Add Oscar Piastri to the 2025 season projections
oscar_piastri = pd.DataFrame({
    'Year': [2025],
    'Driver': ['Oscar Piastri'],
    'Driver_Experience': [3],
    'Avg_Qualifying_Position': [4.2],
    'Avg_Race_Position': [4.5],
    'Pit_Stops': [1.9],
    'Fastest_Lap_Times': [82.5],
    'Driver_Points': [None]
})

# Combine existing drivers with new additions
drivers_2025 = pd.concat([drivers_2025, lando_norris, oscar_piastri], ignore_index=True)

# Prepare feature matrix for 2025 prediction
X_2025 = drivers_2025[['Driver_Experience', 'Avg_Qualifying_Position',
                       'Avg_Race_Position', 'Pit_Stops', 'Fastest_Lap_Times']]

# Generate points predictions for 2025
predicted_points_2025 = model.predict(X_2025)

# Enforce non-negative points
predicted_points_2025 = np.maximum(predicted_points_2025, 0)

# Add predictions to the 2025 dataframe and round to integers
drivers_2025['Predicted_Points'] = np.round(predicted_points_2025).astype(int)

# Sort by predicted points to generate championship standings
drivers_2025 = drivers_2025.sort_values('Predicted_Points', ascending=False).reset_index(drop=True)

# Display 2025 championship prediction results
print("\n2025 F1 Championship Predictions:")
print(drivers_2025[['Driver', 'Driver_Experience', 'Avg_Qualifying_Position',
                    'Avg_Race_Position', 'Pit_Stops', 'Fastest_Lap_Times', 'Predicted_Points']])

# Save predictions to CSV
drivers_2025.to_csv('results/2025_predictions.csv', index=False)

# Highlight the predicted champion and key drivers
champion = drivers_2025.iloc[0]['Driver']
champion_points = drivers_2025.iloc[0]['Predicted_Points']
print(f"\nPredicted 2025 F1 Champion: {champion} with {champion_points} points")

# Highlight key drivers' predicted positions
for driver in ['Lando Norris', 'Oscar Piastri', 'Max Verstappen', 'Lewis Hamilton']:
    driver_row = drivers_2025[drivers_2025['Driver'] == driver]
    if not driver_row.empty:
        position = driver_row.index[0] + 1  # Add 1 because index is 0-based
        points = driver_row['Predicted_Points'].values[0]
        print(f"{driver}'s predicted position: {position}th with {points} points")

print("\nGenerating championship visualization...")
# Visualize 2025 championship predictions
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(drivers_2025)))  # Color gradient for drivers
ax = sns.barplot(x='Driver', y='Predicted_Points', data=drivers_2025, palette=colors)
plt.title('Predicted F1 Driver Points for 2025 Season', fontsize=16)
plt.xlabel('Driver', fontsize=14)
plt.ylabel('Predicted Points', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
plt.grid(True, alpha=0.3)

# Highlight the predicted champion with a gold bar
champion_idx = drivers_2025.index[drivers_2025['Driver'] == champion][0]
ax.patches[champion_idx].set_facecolor('gold')

plt.tight_layout()
plt.savefig('plots/2025_championship_predictions.png')
plt.close()

# Create final standings table
drivers_2025['Rank'] = range(1, len(drivers_2025) + 1)
prediction_table = drivers_2025[['Rank', 'Driver', 'Predicted_Points']]
prediction_table = prediction_table.rename(columns={'Predicted_Points': 'Points'})

print("\nFinal 2025 F1 Championship Prediction Table:")
print(prediction_table)

# Save prediction table to CSV
prediction_table.to_csv('results/2025_championship_standings.csv', index=False)

# Display model equation
print("\nPrediction Model Equation:")
equation = f"Points = {model.intercept_:.2f}"
for i, feature in enumerate(X.columns):
    equation += f" + ({model.coef_[i]:.2f} × {feature})"
print(equation)

print("\nPrediction Complete! Results saved to 'results/' directory and visualization to 'plots/' directory.")