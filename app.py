# ============================================================
# Этап 4. Streamlit-приложение
# Запуск:  streamlit run app.py
# Требует запущенный API (uvicorn api:app --port 8000).
# ============================================================

import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"
DATA_PATH = "real_estate_data.csv"

st.set_page_config(page_title="Прогноз стоимости недвижимости", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[(df["price_currency"] == "TRY") & df["price"].notna() & (df["price"] > 0)]
    return df


@st.cache_data(ttl=60)
def get_model_info():
    r = requests.get(f"{API_URL}/model/info", timeout=10)
    r.raise_for_status()
    return r.json()


tab_form, tab_dashboard, tab_help = st.tabs(["Прогноз цены", "Дашборд", "Справка"])

# ------------------------------------------------------------------
# Вкладка 1. Форма прогноза
# ------------------------------------------------------------------
with tab_form:
    st.header("Прогноз стоимости объекта недвижимости")

    try:
        info = get_model_info()
        sub_types = info["category_values"]["sub_type"]
        heating_types = info["category_values"]["heating_type"]
    except Exception as e:
        st.error(f"Не удалось получить данные от API ({API_URL}). Убедитесь, что API запущен. Ошибка: {e}")
        sub_types, heating_types = ["Daire"], ["Yok"]

    col1, col2 = st.columns(2)
    with col1:
        sub_type = st.selectbox("Тип недвижимости", sub_types)
        heating_type = st.selectbox("Тип отопления", heating_types)
        size = st.number_input("Площадь, м²", min_value=1.0, max_value=5000.0, value=100.0, step=1.0)
        room_count = st.number_input(
            "Количество комнат (напр. 3+1 -> 4)", min_value=0.0, max_value=20.0, value=3.0, step=1.0
        )
    with col2:
        building_age = st.number_input("Возраст здания, лет", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        total_floor_count = st.number_input("Этажность дома", min_value=1.0, max_value=100.0, value=5.0, step=1.0)
        tom = st.number_input("Срок размещения объявления, дней", min_value=0.0, max_value=3650.0, value=30.0, step=1.0)
        listing_days = st.number_input("Длительность объявления, дней", min_value=0.0, max_value=3650.0, value=30.0, step=1.0)

    if st.button("Рассчитать", type="primary"):
        payload = {
            "sub_type": sub_type,
            "heating_type": heating_type,
            "size": size,
            "room_count": room_count,
            "building_age": building_age,
            "total_floor_count": total_floor_count,
            "tom": tom,
            "listing_days": listing_days,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if resp.status_code != 200:
                st.error(f"Ошибка API: {resp.json().get('detail', resp.text)}")
            else:
                result = resp.json()
                st.success(f"Прогнозируемая цена: **{result['predicted_price']:,.0f} TRY**")
                st.caption(
                    f"Диапазон (±MAE модели {result['model_name']}): "
                    f"{result['price_low']:,.0f} — {result['price_high']:,.0f} TRY"
                )
        except requests.exceptions.RequestException as e:
            st.error(f"Не удалось связаться с API по адресу {API_URL}. Убедитесь, что он запущен. Ошибка: {e}")

# ------------------------------------------------------------------
# Вкладка 2. Дашборд
# ------------------------------------------------------------------
with tab_dashboard:
    st.header("Дашборд по датасету и моделям")
    df = load_data()

    c1, c2, c3 = st.columns(3)
    c1.metric("Объявлений (TRY)", f"{len(df):,}")
    c2.metric("Средняя цена", f"{df['price'].mean():,.0f}")
    c3.metric("Медианная цена", f"{df['price'].median():,.0f}")

    st.subheader("Распределение цены (1-99 перцентиль)")
    price_clip = df["price"].clip(upper=df["price"].quantile(0.99))
    st.bar_chart(price_clip.value_counts(bins=40).sort_index())

    st.subheader("Средняя цена по типу недвижимости")
    by_subtype = df.groupby("sub_type")["price"].mean().sort_values(ascending=False)
    st.bar_chart(by_subtype)

    st.subheader("Площадь vs цена")
    scatter_df = df[["size", "price"]].dropna()
    scatter_df = scatter_df[scatter_df["price"] <= scatter_df["price"].quantile(0.99)]
    st.scatter_chart(scatter_df, x="size", y="price")

    st.subheader("Сравнение моделей (метрики на validation)")
    try:
        info = get_model_info()
        metrics_df = pd.DataFrame(info["metrics_validation"]).set_index("model")
        st.dataframe(metrics_df.style.format("{:.2f}"))
        st.caption(f"Лучшая модель: **{info['best_model_name']}**")
        st.bar_chart(metrics_df[["MAE", "RMSE"]])
    except Exception as e:
        st.warning(f"Не удалось получить метрики моделей от API. Ошибка: {e}")

# ------------------------------------------------------------------
# Вкладка 3. Справка
# ------------------------------------------------------------------
with tab_help:
    st.header("Справка")
    st.markdown(
        """
### Что делает приложение
Приложение прогнозирует рыночную стоимость объекта недвижимости (объявления
о **продаже**) на основе его характеристик, а также показывает статистику
по датасету и сравнение обученных моделей.

### Как пользоваться
1. На вкладке **«Прогноз цены»** заполните характеристики объекта и нажмите «Рассчитать».
2. На вкладке **«Дашборд»** можно посмотреть общую статистику по датасету и метрики моделей.

### Описание входных полей
- **Тип недвижимости** — категория объекта (квартира, вилла и т.д.).
- **Тип отопления** — способ отопления объекта.
- **Площадь** — общая площадь объекта, м².
- **Количество комнат** — суммарное число комнат (например, «3+1» → 4).
- **Возраст здания** — возраст здания, лет.
- **Этажность дома** — количество этажей в доме.
- **Срок размещения объявления** — сколько дней объявление уже размещено (tom).
- **Длительность объявления** — сколько дней объявление находилось в публикации.

### О модели
Используется лучшая из нескольких протестированных моделей регрессии
(линейная, дерево решений, случайный лес, градиентный бустинг), выбранная
по средней абсолютной ошибке (MAE) на отложенной выборке. Модель обучена
только на объявлениях о **продаже** в валюте **TRY**, после удаления
экстремальных выбросов цены. Прогноз указывается с диапазоном ±MAE — это
не гарантированная точность, а ориентир, основанный на исторических данных.

**Ограничения:** модель не учитывает конкретный адрес/район, состояние
ремонта и рыночную конъюнктуру на момент запроса — она основана на
историческом наборе объявлений.

### О приложении
Версия 1.0 · Учебный проект TeamWorkMOIBD · Backend: FastAPI, Frontend: Streamlit
"""
    )
