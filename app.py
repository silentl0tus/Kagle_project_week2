import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
# from sklearn.base import BaseEstimator, TransformerMixin
import sklearn
sklearn.set_config(transform_output="pandas")


# Добавь это в блок функций в app.py
def log_transform(X):
    return np.log1p(X)

# Также убедись, что на всякий случай добавлена bool_to_int БЕЗ отступов
def bool_to_int(X):
    return X.astype(int)

# 2. ВСЕ кастомные классы из ноутбука
class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """
    Заполняет пропуски в целевом столбце медианой по группам.
    """
    def __init__(self, group_col, target_col):
        self.group_col = group_col
        self.target_col = target_col
        self.medians_ = None   # словарь или Series с медианами для каждой группы

    def fit(self, X, y=None):
        # X должен быть DataFrame, содержащим оба столбца
        # Вычисляем медиану для каждой группы только на обучающих данных
        self.medians_ = X.groupby(self.group_col)[self.target_col].median()
        return self

    def transform(self, X):
        X = X.copy()
        # Для каждого значения группы подставляем соответствующую медиану
        # Если в X встречается группа, которой не было в fit, используем общую медиану (опционально)
        for group, median in self.medians_.items():
            mask = (X[self.group_col] == group) & (X[self.target_col].isna())
            X.loc[mask, self.target_col] = median
        
        # Обрабатываем оставшиеся NaN (например, группы, не встречавшиеся в обучении)
        # Можно заполнить глобальной медианой или оставить как есть
        if X[self.target_col].isna().any():
            global_median = X[self.target_col].median()
            X[self.target_col].fillna(global_median, inplace=True)
        return X
    
class Ordinal_mapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        for col, mapper in self.mapping.items():
            if col in X.columns:
                X[col] = X[col].map(mapper).fillna(0)
        return X
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Основная логика из твоего ноутбука
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']
        X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
        X['IsRemodeled'] = (X['YearBuilt'] != X['YearRemodAdd']).astype(int)
        X['TotalSF'] = X['GrLivArea'] + X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        X['TotalPorchSF'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch'] + X['WoodDeckSF']
        X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath'] + X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath']
        
        X['HasFireplace'] = (X['Fireplaces'] > 0).astype(int)
        X['HasGarage'] = (X['GarageArea'] > 0).astype(int)
        X['HasPorch'] = (X['TotalPorchSF'] > 0).astype(int)
        
        X['QualSF'] = X['OverallQual'] * X['GrLivArea']
        X['QualTotalSF'] = X['OverallQual'] * X['TotalSF']
        
        return X
    

# 3. Только ПОСЛЕ этого функция загрузки
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "models/voting_pipeline.pkl")
    model = joblib.load(model_path)
    train = pd.read_csv(os.path.join(base_path, "data/train.csv"))
    return model, train

model, train = load_assets()

# 3. ЗАПУСК
model, train = load_assets()

st.title("Прогноз стоимости недвижимости (Ames Housing)")

# Формируем интерфейс
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Основные характеристики")
    overall_qual = st.slider("Общее качество (OverallQual)", 1, 10, 5)
    gr_liv_area = st.number_input("Жилая площадь, кв.фут (GrLivArea)", 300, 6000, 1500)
    year_built = st.number_input("Год постройки", 1872, 2010, 2000)

with col2:
    st.subheader("Детали строения")
    total_bsmt_sf = st.number_input("Площадь подвала (TotalBsmtSF)", 0, 6000, 1000)
    garage_cars = st.selectbox("Мест в гараже", [0, 1, 2, 3, 4], index=2)
    full_bath = st.slider("Полноценные санузлы", 0, 4, 2)

with col3:
    st.subheader("Окружение")
    neighborhood = st.selectbox("Район (Neighborhood)", sorted(train['Neighborhood'].unique()))
    ms_zoning = st.selectbox("Зонирование (MSZoning)", train['MSZoning'].unique())

if st.button("Рассчитать стоимость", type="primary"):
    # Создаем DataFrame с одной строкой, заполненной медианами/модами из train
    input_data = train.drop(columns=['SalePrice']).iloc[[0]].copy()
    
    # Заполняем типичными значениями, чтобы не было ошибок в Pipeline
    for col in input_data.columns:
        if train[col].dtype == 'object':
            input_data[col] = train[col].mode()[0]
        else:
            input_data[col] = train[col].median()

    # Обновляем значениями из интерфейса
    input_data['OverallQual'] = overall_qual
    input_data['GrLivArea'] = gr_liv_area
    input_data['YearBuilt'] = year_built
    input_data['TotalBsmtSF'] = total_bsmt_sf
    input_data['GarageCars'] = garage_cars
    input_data['FullBath'] = full_bath
    input_data['Neighborhood'] = neighborhood
    input_data['MSZoning'] = ms_zoning
    
    # Важно: Добавляем вычисляемые признаки, если они были в обучении
    input_data['HouseAge'] = 2010 - input_data['YearBuilt']
    input_data['TotalSF'] = input_data['GrLivArea'] + input_data['TotalBsmtSF']

    # Предсказание
    try:
        log_prediction = model.predict(input_data)
        final_price = np.expm1(log_prediction)[0]
        
        st.success(f"### Рекомендуемая цена продажи: ${final_price:,.2f}")
    except Exception as e:
        st.error(f"Ошибка при расчете: {e}")
        st.info("Проверьте, что в input_data присутствуют все колонки, на которых училась модель.")

st.sidebar.info("Инструмент для оценки портфеля недвижимости на базе ML-модели.")