import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Wczytanie danych
file_path = 'CollegeDistance.csv'
data = pd.read_csv(file_path)

df = pd.DataFrame(data)

# Kategoryzacja
df_original_size = df.shape[0]
original_columns = df.shape[1]
categorical_columns = ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'income', 'region']

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

new_columns = df.shape[1]
categorical_percentage = ((new_columns - original_columns) / original_columns) * 100
print(f"Procent nowych kolumn po kategoryzacji: {categorical_percentage:.2f}%")


# Standaryzacja
numeric_cols = ['score', 'unemp', 'wage', 'distance', 'tuition', 'education']
scaler_standard = StandardScaler()

df_standardized = df.copy()
df_standardized[numeric_cols] = scaler_standard.fit_transform(df[numeric_cols])

numerical_modified_count = len(numeric_cols) * df.shape[0]
numerical_percentage = (numerical_modified_count / (df_original_size * len(df.columns))) * 100
print(f"Procent danych zmodyfikowanych przez standaryzację: {numerical_percentage:.2f}%")

print("Standaryzacja i kategoryzacja zakończone.")

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df_standardized.drop(columns=['score']), df_standardized['score'], test_size=0.2, random_state=42)

print("Dane zostały podzielone na zbiór treningowy i testowy.")

# Obliczenie liczby danych w zbiorze treningowym i testowym
print(f"Liczba danych w zbiorze treningowym: {X_train.shape[0]}")
print(f"Liczba danych w zbiorze testowym: {X_test.shape[0]}")

print("\nRozpoczyanie uczenia\n")

# Lista modeli do przetestowania
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Słownik do przechowywania wyników
results = {}

# Trenowanie, przewidywanie i ocena każdego modelu
for model_name, model in models.items():
    # Trenowanie modelu
    model.fit(X_train, y_train)

    # Przewidywanie na zbiorze testowym
    y_pred = model.predict(X_test)

    # Obliczanie metryk
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Przechowywanie wyników
    results[model_name] = {'MSE': mse, 'R2 Score': r2}

# Wyświetlanie wyników
results_df = pd.DataFrame(results).T
print("Wyniki modeli:")
print(results_df)

# Wybór najlepszego modelu
best_model_name = results_df['MSE'].idxmin()
print(f"\nNajlepszy model: {best_model_name}")
print(f"Metryki najlepszego modelu:\n{results_df.loc[best_model_name]}")
