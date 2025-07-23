import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Настройки страницы
st.set_page_config(page_title="🐧 Penguin Classifier", layout="wide")
st.title('🐧 Penguin Classifier - Обучение и предсказание')
st.write("## Работа с датасетом пингвинов")

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

df = load_data()

# Показываем случайные 10 строк
st.subheader("🔎 Случайные 10 строк")
st.dataframe(df.sample(10), use_container_width=True)

# Визуализации
st.subheader("📊 Визуализация данных")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x="species", color="island", barmode="group", title="Распределение видов по островам")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.scatter(df, x="bill_length_mm", y="flipper_length_mm", color="species", title="Длина клюва vs Длина крыла")
    st.plotly_chart(fig2, use_container_width=True)

# Target Mean Encoding
st.subheader("🧠 Обучение моделей")
df_encoded = df.copy()
for col in ['island', 'sex']:
    means = df_encoded.groupby(col)['species'].apply(lambda x: x.map({k: i for i, k in enumerate(x.unique())}).mean())
    df_encoded[col + '_mean'] = df_encoded[col].map(means)
df_encoded.drop(columns=['island', 'sex'], inplace=True)

# Train/Test Split
X = df_encoded.drop(columns=['species'])
y = df_encoded['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение моделей
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))
    results.append({
        'Model': name,
        'Train Accuracy': round(acc_train, 2),
        'Test Accuracy': round(acc_test, 2)
    })

st.write("### 📋 Сравнение моделей по точности")
st.table(pd.DataFrame(results))

# Сайдбар — ввод параметров
st.sidebar.header("🔮 Предсказание по параметрам")

island_input = st.sidebar.selectbox("Остров", df['island'].unique())
sex_input = st.sidebar.selectbox("Пол", df['sex'].unique())
bill_length = st.sidebar.slider("Длина клюва (мм)", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
bill_depth = st.sidebar.slider("Глубина клюва (мм)", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
flipper_length = st.sidebar.slider("Длина крыла (мм)", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
body_mass = st.sidebar.slider("Масса тела (г)", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()), float(df['body_mass_g'].mean()))

# Target mean encoding на новые данные
def encode_feature(value, colname):
    mapping = df.groupby(colname)['species'].apply(lambda x: x.map({k: i for i, k in enumerate(x.unique())}).mean())
    return mapping.get(value, 0.0)

island_mean = encode_feature(island_input, 'island')
sex_mean = encode_feature(sex_input, 'sex')

user_data = pd.DataFrame([{
    'bill_length_mm': bill_length,
    'bill_depth_mm': bill_depth,
    'flipper_length_mm': flipper_length,
    'body_mass_g': body_mass,
    'island_mean': island_mean,
    'sex_mean': sex_mean
}])

# Предсказания моделей
st.sidebar.subheader("Результат предсказания:")
for name, model in models.items():
    prediction = model.predict(user_data)[0]
    st.sidebar.write(f"**{name} предсказал:** {prediction}")
