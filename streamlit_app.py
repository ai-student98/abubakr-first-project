import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="🐧 Penguin Classifier", layout="wide")
st.title('🐧 Penguin Classifier - Обучение и предсказание')
st.write("## Работа с датасетом пингвинов")

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

st.subheader("🔎 Случайные 10 строк")
st.dataframe(df.sample(10), use_container_width=True)

st.subheader("📊 Визуализация данных")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x="species", color="island", barmode="group", title="Распределение видов по островам")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.scatter(df, x="bill_length_mm", y="flipper_length_mm", color="species", title="Длина клюва vs Длина крыла")
    st.plotly_chart(fig2, use_container_width=True)

X_raw = df.drop(columns=["species"])
y = df["species"]

y_numeric, classes = pd.factorize(y)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
y_train_numeric = pd.Series(pd.factorize(y_train)[0], index=y_train.index)

# TME вручную
means_island = X_train_raw.join(y_train_numeric).groupby("island")[0].mean()
means_sex = X_train_raw.join(y_train_numeric).groupby("sex")[0].mean()

X_train = X_train_raw.copy()
X_train["island_mean"] = X_train["island"].map(means_island).fillna(y_train_numeric.mean())
X_train["sex_mean"] = X_train["sex"].map(means_sex).fillna(y_train_numeric.mean())
X_train = X_train.drop(columns=["island", "sex"])

X_test = X_test_raw.copy()
X_test["island_mean"] = X_test["island"].map(means_island).fillna(y_train_numeric.mean())
X_test["sex_mean"] = X_test["sex"].map(means_sex).fillna(y_train_numeric.mean())
X_test = X_test.drop(columns=["island", "sex"])

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

st.sidebar.header("🔮 Предсказание по параметрам")

island_input = st.sidebar.selectbox("Остров", df['island'].unique())
sex_input = st.sidebar.selectbox("Пол", df['sex'].unique())
bill_length = st.sidebar.slider("Длина клюва (мм)", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
bill_depth = st.sidebar.slider("Глубина клюва (мм)", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
flipper_length = st.sidebar.slider("Длина крыла (мм)", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
body_mass = st.sidebar.slider("Масса тела (г)", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()), float(df['body_mass_g'].mean()))

user_df = pd.DataFrame([{
    'island': island_input,
    'sex': sex_input,
    'bill_length_mm': bill_length,
    'bill_depth_mm': bill_depth,
    'flipper_length_mm': flipper_length,
    'body_mass_g': body_mass
}])

user_df["island_mean"] = user_df["island"].map(means_island).fillna(y_train_numeric.mean())
user_df["sex_mean"] = user_df["sex"].map(means_sex).fillna(y_train_numeric.mean())
user_encoded = user_df.drop(columns=["island", "sex"])

st.sidebar.subheader("📈 Результаты предсказания")
for name, model in models.items():
    pred_class = model.predict(user_encoded)[0]
    pred_proba = model.predict_proba(user_encoded)[0]
    proba_df = pd.DataFrame({'Вид': model.classes_, 'Вероятность': pred_proba})
    st.sidebar.markdown(f"**{name}: {pred_class}**")
    st.sidebar.dataframe(proba_df.set_index("Вид"), use_container_width=True)
