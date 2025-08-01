import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import category_encoders as ce
import plotly.express as px


# --- Конфигурация страницы Streamlit ---
st.set_page_config(page_title="🚢 Прогноз выживания на Титанике", layout="wide")
st.title('🚢 Прогноз выживания на Титанике - Обучение и предсказание')
st.write('## Работа с датасетом Титаника')

# Загрузка и предобработка данных
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("titanic.csv")
    
    # Предобработка данных
    df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    return df

df = load_and_preprocess_data()

st.subheader("🔍 Случайные 10 строк")
st.dataframe(df.sample(10), use_container_width=True)

st.subheader("📊 Визуализация данных")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x="Survived", color="Sex", barmode="group",
                        title="Выживание по полу")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df, x="Age", color="Survived", marginal="rug",
                        title="Распределение возраста по выживанию")
    st.plotly_chart(fig2, use_container_width=True)

# ОПРЕДЕЛЕНИЕ ПРИЗНАКОВ И ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
X = df.drop(['Survived'], axis=1)
y = df['Survived']

# РАЗДЕЛЕНИЕ ДАННЫХ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
categorical_cols = ['Pclass', 'Sex', 'Embarked']
encoder = ce.TargetEncoder(cols=categorical_cols)

X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

# Использование RandomForestClassifier, как было указано в задании
model = RandomForestClassifier(n_estimators=100, random_state=42)

# ОБУЧЕНИЕ МОДЕЛИ
model.fit(X_train_encoded, y_train)

# ПРЕДСКАЗАНИЯ И ОЦЕНКА ТОЧНОСТИ
acc_train = accuracy_score(y_train, model.predict(X_train_encoded))
acc_test = accuracy_score(y_test, model.predict(X_test_encoded))      
    
results = pd.DataFrame([{
    'Model': 'RandomForestClassifier',
    'Train Accuracy': round(acc_train, 2),
    'Test Accuracy': round(acc_test, 2)
}])

st.write("### 📊 Сравнение моделей по точности")
st.table(results)
