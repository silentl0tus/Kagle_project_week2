import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Предсказание стоимости дома 🏠")

# 1. Загружаем модель
model = joblib.load('house_model.pkl')

# 2. Нам нужны названия всех колонок, которые были при обучении.
# Самый простой способ — загрузить одну строку из train.csv как шаблон
@st.cache_data
def get_template():
    df = pd.read_csv('./data/train.csv').drop('SalePrice', axis=1)
    return df.iloc[0:1].copy() # Берем первую строку как образец

template_df = get_template()

st.subheader("Измените ключевые параметры:")

# Создаем интерфейс (выводим только самое важное)
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Общее качество (1-10)", 1, 10, int(template_df['OverallQual'].iloc[0]))
    gr_liv_area = st.number_input("Жилая площадь (кв. футы)", value=int(template_df['GrLivArea'].iloc[0]))
    year_built = st.number_input("Год постройки", value=int(template_df['YearBuilt'].iloc[0]))

with col2:
    total_bsmt_sf = st.number_input("Площадь подвала", value=int(template_df['TotalBsmtSF'].iloc[0]))
    garage_cars = st.slider("Мест в гараже", 0, 4, int(template_df['GarageCars'].iloc[0]))
    lot_area = st.number_input("Площадь участка", value=int(template_df['LotArea'].iloc[0]))

# Кнопка расчета
if st.button("Рассчитать стоимость"):
    # СОЗДАЕМ ДАННЫЕ ДЛЯ МОДЕЛИ
    # Берем наш шаблон (в котором есть все 80 колонок)
    input_data = template_df.copy()
    
    # Обновляем в нем только те поля, которые ввел пользователь
    input_data['OverallQual'] = overall_qual
    input_data['GrLivArea'] = gr_liv_area
    input_data['YearBuilt'] = year_built
    input_data['TotalBsmtSF'] = total_bsmt_sf
    input_data['GarageCars'] = garage_cars
    input_data['LotArea'] = lot_area
    
    # Добавляем расчетные признаки (те, что ты делал во второй день)
    # Важно: расчет должен быть идентичен тому, что в ноутбуке!
    input_data['TotalSF'] = input_data['TotalBsmtSF'] + input_data['1stFlrSF'] + input_data['2ndFlrSF']
    input_data['HouseAge'] = 2010 - input_data['YearBuilt'] # 2010 - примерный год отсечки данных
    
    # Делаем предсказание
    try:
        prediction_log = model.predict(input_data)
        prediction = np.expm1(prediction_log)
        
        st.success(f"Предполагаемая цена: ${prediction[0]:,.2f}")
        st.balloons()
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")