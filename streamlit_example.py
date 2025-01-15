import streamlit as st
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit.components.v1 as components  # Для вставки HTML

# Заголовок приложения
st.title("Прогноз качества вина")
st.write("""
Введите химические характеристики вина, чтобы получить прогноз качества.

""")

# Загрузка данных
@st.cache_data
def load_data():
    data = pd.read_csv("winequality.csv")  # Убедитесь, что файл winequality.csv находится в той же папке
    data['type'] = data['type'].map({'white': 1, 'red': 0})  # Кодируем тип вина: white = 1, red = 0
    X = data.drop("quality", axis=1)
    y = data["quality"]
    return X, y

X, y = load_data()

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели с заданными параметрами
model = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, n_estimators=200, random_state=42)
model.fit(X_train, y_train)



# Функция для Bar Plot
def plot_shap_bar(input_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    st.subheader("Bar Plot:")
    plt.figure()
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.clf()

# Функция для Waterfall Plot
def plot_shap_waterfall(input_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)
    st.subheader("Waterfall Plot:")
    plt.figure()
    shap.plots.waterfall(shap_values[0])
    st.pyplot(plt.gcf())
    plt.clf()

# Интерфейс для ввода характеристик вина
st.sidebar.header("Характеристики вина:")

fixed_acidity = st.sidebar.slider("Fixed Acidity", float(X["fixed acidity"].min()), float(X["fixed acidity"].max()), float(X["fixed acidity"].mean()))
volatile_acidity = st.sidebar.slider("Volatile Acidity", float(X["volatile acidity"].min()), float(X["volatile acidity"].max()), float(X["volatile acidity"].mean()))
citric_acid = st.sidebar.slider("Citric Acid", float(X["citric acid"].min()), float(X["citric acid"].max()), float(X["citric acid"].mean()))
residual_sugar = st.sidebar.slider("Residual Sugar", float(X["residual sugar"].min()), float(X["residual sugar"].max()), float(X["residual sugar"].mean()))
chlorides = st.sidebar.slider("Chlorides", float(X["chlorides"].min()), float(X["chlorides"].max()), float(X["chlorides"].mean()))
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", float(X["free sulfur dioxide"].min()), float(X["free sulfur dioxide"].max()), float(X["free sulfur dioxide"].mean()))
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", float(X["total sulfur dioxide"].min()), float(X["total sulfur dioxide"].max()), float(X["total sulfur dioxide"].mean()))
density = st.sidebar.slider("Density", float(X["density"].min()), float(X["density"].max()), float(X["density"].mean()))
pH = st.sidebar.slider("pH", float(X["pH"].min()), float(X["pH"].max()), float(X["pH"].mean()))
sulphates = st.sidebar.slider("Sulphates", float(X["sulphates"].min()), float(X["sulphates"].max()), float(X["sulphates"].mean()))
alcohol = st.sidebar.slider("Alcohol", float(X["alcohol"].min()), float(X["alcohol"].max()), float(X["alcohol"].mean()))
type_white = st.sidebar.selectbox("Type", [0, 1], format_func=lambda x: "Белое" if x == 1 else "Красное")

# данные для прогноза
input_data = pd.DataFrame({
    "fixed acidity": [fixed_acidity],
    "volatile acidity": [volatile_acidity],
    "citric acid": [citric_acid],
    "residual sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free sulfur dioxide": [free_sulfur_dioxide],
    "total sulfur dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol],
    "type": [type_white]
})

# Кнопка для выполнения прогноза
if st.button("Прогнозировать"):
    prediction = model.predict(input_data)[0]
    st.session_state.prediction = prediction
    st.session_state.input_data = input_data
    st.session_state.show_visualization = True

# Если прогноз уже выполнен, показать результат
if st.session_state.get("show_visualization", False):
    st.subheader(f"Прогнозируемое качество вина: {st.session_state.prediction:.2f}")

    # Выбор типа графика
    visualization_type = st.selectbox(
        "Выберите тип визуализации:",
        ["Bar Plot", "Waterfall Plot"]
    )

    # Генерация выбранного графика
    if visualization_type == "Bar Plot":
        plot_shap_bar(st.session_state.input_data)
    elif visualization_type == "Waterfall Plot":
        plot_shap_waterfall(st.session_state.input_data)