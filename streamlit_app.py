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


df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# --- Предобработка данных ---
# Заполняем пропуски в 'Age' медианным значением
df['Age'].fillna(df['Age'].median(), inplace=True)
# Заполняем пропуски в 'Embarked' самым частым значением (модой)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# Преобразуем 'Survived' в более понятные метки для графиков
df['SurvivalStatus'] = df['Survived'].map({0: 'Не выжил', 1: 'Выжил'})


st.subheader("🔍 Случайные 10 строк")
st.dataframe(df.sample(10), use_container_width=True)

st.subheader("📊 Визуализация данных")
col1, col2 = st.columns(2)

with col1:
    # Распределение выживших по классу билета
    fig1 = px.histogram(df, x="Pclass", color="SurvivalStatus", barmode="group",
                        title="Распределение выживших по классу билета",
                        labels={'Pclass': 'Класс билета', 'SurvivalStatus': 'Статус выживания'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Зависимость выживаемости от возраста и пола
    fig2 = px.scatter(df, x="Age", y="Fare", color="SurvivalStatus",
                      title="Стоимость билета vs Возраст",
                      labels={'Age': 'Возраст', 'Fare': 'Стоимость билета', 'SurvivalStatus': 'Статус выживания'})
    st.plotly_chart(fig2, use_container_width=True)

# --- Подготовка данных для обучения ---
# Удаляем ненужные столбцы и целевую переменную
X = df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'SurvivalStatus'], axis=1)
y = df['Survived']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- Кодирование категориальных признаков ---
# Используем TargetEncoder для 'Sex' и 'Embarked'
encoder = ce.TargetEncoder(cols=['Sex', 'Embarked'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

# --- Обучение и оценка моделей ---
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = []
for name, model in models.items():
    # Обучение модели
    model.fit(X_train_encoded, y_train)

    # Предсказания и оценка точности
    acc_train = accuracy_score(y_train, model.predict(X_train_encoded))
    acc_test = accuracy_score(y_test, model.predict(X_test_encoded))

    results.append({
        'Model': name,
        'Train Accuracy': round(acc_train, 2),
        'Test Accuracy': round(acc_test, 2)
    })

st.write("### 📊 Сравнение моделей по точности")
st.table(pd.DataFrame(results))

# --- Интерфейс для предсказания в боковой панели ---
st.sidebar.header("🔮 Предсказание по параметрам")

# Виджеты для ввода данных пользователем
pclass_input = st.sidebar.selectbox("Класс билета", sorted(df['Pclass'].unique()))
sex_input = st.sidebar.selectbox("Пол", df['Sex'].unique())
age_input = st.sidebar.slider("Возраст", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))
sibsp_input = st.sidebar.slider("Кол-во братьев/сестер/супругов на борту", int(df['SibSp'].min()), int(df['SibSp'].max()), int(df['SibSp'].mode()[0]))
parch_input = st.sidebar.slider("Кол-во родителей/детей на борту", int(df['Parch'].min()), int(df['Parch'].max()), int(df['Parch'].mode()[0]))
fare_input = st.sidebar.slider("Стоимость билета", float(df['Fare'].min()), float(df['Fare'].max()), float(df['Fare'].median()))
embarked_input = st.sidebar.selectbox("Порт посадки", df['Embarked'].unique())

# Создание DataFrame из пользовательского ввода
user_input = pd.DataFrame([{
    'Pclass': pclass_input,
    'Sex': sex_input,
    'Age': age_input,
    'SibSp': sibsp_input,
    'Parch': parch_input,
    'Fare': fare_input,
    'Embarked': embarked_input
}])

# Кодирование пользовательского ввода
user_encoded = encoder.transform(user_input)
# Убедимся, что порядок столбцов совпадает с обучающими данными
user_encoded = user_encoded[X_train_encoded.columns]


st.sidebar.subheader("📈 Результаты предсказания")
for name, model in models.items():
    # Получение предсказания и вероятностей
    pred_val = model.predict(user_encoded)[0]
    pred_text = "Выжил" if pred_val == 1 else "Не выжил"
    proba = model.predict_proba(user_encoded)[0]

    st.sidebar.markdown(f"**{name}: {pred_text}**")
    proba_df = pd.DataFrame({
        'Статус': ['Не выжил', 'Выжил'],
        'Вероятность': proba
    })
    st.sidebar.dataframe(proba_df.set_index("Статус"), use_container_width=True)


# st.subheader("🔍 Случайные 10 строк")
# st.dataframe(df.sample(10), use_container_width=True)

# st.subheader("📊 Визуализация данных")
# col1, col2 = st.columns(2)

# @st.cache_data
# def load_and_preprocess_data():
#     df = pd.read_csv("titanic.csv")
    
#     # Предобработка данных
#     df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
#     df['Age'].fillna(df['Age'].median(), inplace=True)
#     df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
#     return df

# df = load_and_preprocess_data()

# st.subheader("🔍 Случайные 10 строк")
# st.dataframe(df.sample(10), use_container_width=True)

# st.subheader("📊 Визуализация данных")
# col1, col2 = st.columns(2)

# with col1:
#     fig1 = px.histogram(df, x="Survived", color="Sex", barmode="group",
#                         title="Выживание по полу")
#     st.plotly_chart(fig1, use_container_width=True)

# with col2:
#     fig2 = px.histogram(df, x="Age", color="Survived", marginal="rug",
#                         title="Распределение возраста по выживанию")
#     st.plotly_chart(fig2, use_container_width=True)

# # ОПРЕДЕЛЕНИЕ ПРИЗНАКОВ И ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# X = df.drop(['Survived'], axis=1)
# y = df['Survived']

# # РАЗДЕЛЕНИЕ ДАННЫХ
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# categorical_cols = ['Pclass', 'Sex', 'Embarked']
# encoder = ce.TargetEncoder(cols=categorical_cols)

# X_train_encoded = encoder.fit_transform(X_train, y_train)
# X_test_encoded = encoder.transform(X_test)

# # Использование RandomForestClassifier, как было указано в задании
# model = RandomForestClassifier(n_estimators=100, random_state=42)

# # ОБУЧЕНИЕ МОДЕЛИ
# model.fit(X_train_encoded, y_train)

# # ПРЕДСКАЗАНИЯ И ОЦЕНКА ТОЧНОСТИ
# acc_train = accuracy_score(y_train, model.predict(X_train_encoded))
# acc_test = accuracy_score(y_test, model.predict(X_test_encoded))      
    
# results = pd.DataFrame([{
#     'Model': 'RandomForestClassifier',
#     'Train Accuracy': round(acc_train, 2),
#     'Test Accuracy': round(acc_test, 2)
# }])

# st.write("### 📊 Сравнение моделей по точности")
# st.table(results)
