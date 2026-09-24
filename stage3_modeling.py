# ============================================================
# Этап 3. Построение модели регрессии
# Продолжение ноутбука TeamWorkMOIBD (после этапа 2).
# Вставить как новые ячейки в конец ноутбука — код этапов 1-2 не трогаем.
# ============================================================

# --- Если ноутбук перезапускали, эта ячейка сама всё пересоберёт заново.
# --- Если X_train_ready/X_val_ready и т.д. уже есть в памяти из этапа 2 —
# --- код всё равно отработает корректно, т.к. просто переопределит те же переменные.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

RANDOM_STATE = 42

# ---------- 3.0 Пересборка данных (совпадает с этапом 1-2, ничего там не меняем) ----------
df = pd.read_csv('real_estate_data.csv', low_memory=False)
df = df[(df['price_currency'] == 'TRY') & df['price'].notna() & (df['price'] > 0)].copy()

# ВАЖНО: listing_type смешивает объявления о продаже (1), долгосрочной (2) и
# посуточной (3) аренде — это принципиально разные шкалы цены (сотни тысяч vs
# тысячи vs сотни), и как единую регрессионную задачу их решать некорректно.
# Это не правка этапов 1-2 (их код не меняем), а решение на этапе 3: строим
# модель прогноза цены ПРОДАЖИ, как и заявлено в теме проекта.
df = df[df['listing_type'] == 1].drop(columns=['listing_type'])

# Убираем экстремальные выбросы цены (1-2% с каждой стороны): в данных
# встречаются записи вроде "1.65 млрд TRY за дом 426 м2" — это, очевидно,
# ошибки ввода, а не реальные сделки. Такой выброс задаёт для всех моделей
# гигантский масштаб ошибки и не даёт им научиться на "нормальных" объектах.
# Это тоже решение уровня этапа 3 (моделирование), а не правка этапа 1-2.
price_low, price_high = df['price'].quantile([0.02, 0.98])
df = df[(df['price'] >= price_low) & (df['price'] <= price_high)].copy()

df['listing_days'] = (pd.to_datetime(df['end_date'], errors='coerce')
                       - pd.to_datetime(df['start_date'], errors='coerce')).dt.days
df = df.drop(columns=['furnished', 'start_date', 'end_date', 'id', 'price_currency', 'type'])

df['size_log'] = np.log1p(df['size'].clip(lower=0))

def room_to_number(x):
    if pd.isna(x):
        return np.nan
    parts = str(x).split('+')
    try:
        return sum(float(p) for p in parts)
    except ValueError:
        return np.nan

df['room_count_num'] = df['room_count'].apply(room_to_number)
df = df.drop(columns=['room_count', 'address'])

def range_to_number(x, cap_marker, cap_value):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if cap_marker in x:
        return cap_value
    if '-' in x:
        try:
            a, b = x.split('-')
            return (float(a) + float(b.split()[0])) / 2
        except ValueError:
            return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan

df['building_age_num'] = df['building_age'].apply(lambda x: range_to_number(x, '40 ve', 40))
df['total_floor_num'] = df['total_floor_count'].apply(lambda x: range_to_number(x, '20 ve', 20))
df = df.drop(columns=['building_age', 'total_floor_count', 'floor_no'])

X = df.drop(columns=['price'])
y = df['price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                       ('scaler', StandardScaler())]), numeric_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                       ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols),
])

X_train_ready = preprocessor.fit_transform(X_train)
X_val_ready = preprocessor.transform(X_val)
X_test_ready = preprocessor.transform(X_test)

print('Train/val/test:', X_train_ready.shape, X_val_ready.shape, X_test_ready.shape)

# ---------- 3.1-3.2 Модели и подбор гиперпараметров ----------
# 4 модели разной природы: линейная, дерево, бэггинг (RandomForest), бустинг (GradientBoosting)

model_grids = {
    'Ridge': (
        Ridge(random_state=RANDOM_STATE),
        {'alpha': [0.1, 1.0, 5.0, 10.0, 50.0]},
        'grid'
    ),
    'DecisionTree': (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {'max_depth': [4, 6, 8, 10, 12, None], 'min_samples_leaf': [1, 5, 10]},
        'grid'
    ),
    'RandomForest': (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {'n_estimators': [100, 200, 300], 'max_depth': [8, 12, 16, None],
         'min_samples_leaf': [1, 2, 5]},
        'random'
    ),
    'GradientBoosting': (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {'n_estimators': [100, 200, 300], 'max_depth': [2, 3, 4],
         'learning_rate': [0.03, 0.05, 0.1]},
        'random'
    ),
}

fitted_models = {}
for name, (estimator, grid, search_type) in model_grids.items():
    print(f'\nПодбор гиперпараметров: {name}...')
    if search_type == 'grid':
        search = GridSearchCV(estimator, grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    else:
        search = RandomizedSearchCV(estimator, grid, n_iter=10, cv=5,
                                     scoring='neg_mean_absolute_error',
                                     random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_train_ready, y_train)
    fitted_models[name] = search.best_estimator_
    print(f'{name}: лучшие параметры = {search.best_params_}')

# ---------- 3.3 Метрики качества (на validation) ----------
def compute_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'R2': r2_score(y_true, y_pred),
    }

results = []
for name, model in fitted_models.items():
    val_pred = model.predict(X_val_ready)
    metrics = compute_metrics(y_val, val_pred)
    metrics['model'] = name
    results.append(metrics)

results_df = pd.DataFrame(results).set_index('model')[['MAE', 'RMSE', 'MAPE', 'R2']]
print('\nСравнение моделей на validation:')
display(results_df)

# ---------- 3.4 Сравнение моделей (графики) ----------
results_df[['MAE', 'RMSE']].plot(kind='bar', figsize=(8, 5), title='MAE / RMSE по моделям (validation)')
plt.ylabel('Ошибка, TRY')
plt.xticks(rotation=0)
plt.show()

results_df['R2'].plot(kind='bar', figsize=(7, 4), title='R2 по моделям (validation)', color='seagreen')
plt.xticks(rotation=0)
plt.show()

# ---------- 3.5 Выбор финальной модели ----------
best_model_name = results_df['MAE'].idxmin()
best_model = fitted_models[best_model_name]
print(f'\nЛучшая модель по MAE на validation: {best_model_name}')

# Предсказание против факта + остатки для лучшей модели (на test)
best_test_pred = best_model.predict(X_test_ready)
test_metrics = compute_metrics(y_test, best_test_pred)
print(f'Метрики {best_model_name} на test:', test_metrics)

plt.figure(figsize=(6, 6))
lims = [0, min(y_test.max(), best_test_pred.max())]
plt.scatter(y_test, best_test_pred, alpha=0.3, s=10)
plt.plot(lims, lims, 'r--')
plt.xlabel('Факт, TRY')
plt.ylabel('Прогноз, TRY')
plt.title(f'Прогноз vs факт — {best_model_name} (test)')
plt.show()

residuals = y_test.values - best_test_pred
plt.figure(figsize=(8, 4))
plt.scatter(best_test_pred, residuals, alpha=0.3, s=10)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Прогноз, TRY')
plt.ylabel('Остаток (факт - прогноз)')
plt.title(f'Остатки — {best_model_name} (test)')
plt.show()

# Важность признаков / коэффициенты
feature_names = (numeric_cols +
                  list(preprocessor.named_transformers_['cat']
                       .named_steps['encoder'].get_feature_names_out(categorical_cols)))

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    importances.sort_values().plot(kind='barh', figsize=(8, 6), title=f'Важность признаков — {best_model_name}')
    plt.show()
elif hasattr(best_model, 'coef_'):
    coefs = pd.Series(best_model.coef_, index=feature_names).sort_values(key=abs, ascending=False).head(15)
    coefs.sort_values().plot(kind='barh', figsize=(8, 6), title=f'Коэффициенты — {best_model_name}')
    plt.show()

# ---------- Сохранение модели и препроцессора для приложения (Этап 4) ----------
# Финальный пайплайн: сырые признаки -> preprocessor -> лучшая модель.
# Обучаем на train+val, чтобы приложение использовало максимум доступных данных.
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model),
])
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
final_pipeline.fit(X_trainval, y_trainval)

artifact = {
    'pipeline': final_pipeline,
    'best_model_name': best_model_name,
    'feature_columns': X.columns.tolist(),      # сырые признаки, которые ждёт пайплайн
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'category_values': {c: sorted(X[c].dropna().unique().tolist()) for c in categorical_cols},
    'metrics_validation': results_df.reset_index().to_dict(orient='records'),
    'metrics_test_best_model': test_metrics,
    'mae_test': test_metrics['MAE'],            # используем как +-доверительный диапазон в приложении
}

joblib.dump(artifact, 'model_pipeline.joblib')
print('\nМодель и препроцессор сохранены в model_pipeline.joblib')
print('Итоговая таблица метрик (validation):')
display(results_df)
