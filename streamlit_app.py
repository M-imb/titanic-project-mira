import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
import plotly.express as px
import plotly.graph_objects as go # Все еще нужен для Gauge chart

# --- Конфигурация страницы Streamlit ---
st.set_page_config(page_title="🚢 Прогноз выживания на Титанике (Упрощенная)", layout="wide")
st.title('🚢 Прогноз выживания на Титанике - Упрощенная версия')
st.write('## Работа с датасетом Титаника')

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# --- Предобработка данных (только основные пропуски и SurvivalStatus) ---
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['SurvivalStatus'] = df['Survived'].map({0: 'Не выжил', 1: 'Выжил'})

# *** УДАЛЕНЫ FamilySize, IsAlone, Title и их генерация ***

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
    # Зависимость выживаемости от пола
    fig2 = px.histogram(df, x="Sex", color="SurvivalStatus", barmode="group",
                        title="Распределение выживших по полу",
                        labels={'Sex': 'Пол', 'SurvivalStatus': 'Статус выживания'})
    st.plotly_chart(fig2, use_container_width=True)

# *** УДАЛЕНЫ Violin Plot и Sunburst Chart для упрощения ***

# --- Подготовка данных для обучения (для верхней секции "Сравнение моделей по точности") ---
# Используем только базовые признаки
features_for_initial_comparison = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_initial_comparison = df[features_for_initial_comparison]
y_initial_comparison = df['Survived']

X_train_initial_comparison, X_test_initial_comparison, y_train_initial_comparison, y_test_initial_comparison = train_test_split(X_initial_comparison, y_initial_comparison, test_size=0.3, random_state=42, stratify=y_initial_comparison)

# --- Кодирование категориальных признаков для верхней секции ---
encoder_initial_comparison = ce.TargetEncoder(cols=['Sex', 'Embarked'])
X_train_encoded_initial_comparison = encoder_initial_comparison.fit_transform(X_train_initial_comparison, y_train_initial_comparison)
X_test_encoded_initial_comparison = encoder_initial_comparison.transform(X_test_initial_comparison)

# --- Обучение и оценка моделей ---
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train_encoded_initial_comparison, y_train_initial_comparison)
    acc_train = accuracy_score(y_train_initial_comparison, model.predict(X_train_encoded_initial_comparison))
    acc_test = accuracy_score(y_test_initial_comparison, model.predict(X_test_encoded_initial_comparison))
    results.append({
        'Model': name,
        'Train Accuracy': round(acc_train, 2),
        'Test Accuracy': round(acc_test, 2)
    })

st.write("### 📊 Сравнение моделей по точности")
st.table(pd.DataFrame(results))

# --- Интерфейс для предсказания в боковой панели (УПРОЩЕН) ---
st.sidebar.header("🔮 Предсказание по параметрам")

# Оставили только самые важные параметры
pclass_input_sb = st.sidebar.selectbox("Класс билета", sorted(df['Pclass'].unique()), key='sb_pclass_simple')
sex_input_sb = st.sidebar.selectbox("Пол", df['Sex'].unique(), key='sb_sex_simple')
age_input_sb = st.sidebar.slider("Возраст", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()), key='sb_age_simple')
fare_input_sb = st.sidebar.slider("Стоимость билета", float(df['Fare'].min()), float(df['Fare'].max()), float(df['Fare'].median()), key='sb_fare_simple')
embarked_input_sb = st.sidebar.selectbox("Порт посадки", df['Embarked'].unique(), key='sb_embarked_simple')

# Для простоты, SibSp и Parch будут по умолчанию 0 в этом упрощенном варианте
# Если модель требует их, они будут добавлены в DataFrame с нулевыми значениями
user_input_sb = pd.DataFrame([{
    'Pclass': pclass_input_sb,
    'Sex': sex_input_sb,
    'Age': age_input_sb,
    'SibSp': 0,  # Упрощение: задаем 0 по умолчанию
    'Parch': 0,  # Упрощение: задаем 0 по умолчанию
    'Fare': fare_input_sb,
    'Embarked': embarked_input_sb
}])

user_encoded_sb = encoder_initial_comparison.transform(user_input_sb)
user_encoded_sb = user_encoded_sb[X_train_encoded_initial_comparison.columns]

st.sidebar.subheader("📈 Результаты предсказания")
for name, model in models.items():
    pred_val = model.predict(user_encoded_sb)[0]
    pred_text = "Выжил" if pred_val == 1 else "Не выжил"
    proba = model.predict_proba(user_encoded_sb)[0]

    st.sidebar.markdown(f"**{name}: {pred_text}**")
    proba_df = pd.DataFrame({
        'Статус': ['Не выжил', 'Выжил'],
        'Вероятность': proba
    })
    st.sidebar.dataframe(proba_df.set_index("Статус"), use_container_width=True)

# *** УДАЛЕНЫ CSS стили для вкладок ***

# --- Создание вкладок ---
tab1, tab2, tab3 = st.tabs(["📊 Анализ данных", "🤖 Обучение и настройка моделей", "🔮 Сделать прогноз"])

# --- ВКЛАДКА 1: АНАЛИЗ ДАННЫХ (УПРОЩЕН) ---
with tab1:
    st.header("Обзор и визуализация данных")
    st.write("### 📋 Данные о пассажирах")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("### 📈 Интерактивные графики")
    col1, col2 = st.columns(2)
    with col1:
        # Просто показываем гистограмму по Pclass без выбора признака
        fig_pclass = px.histogram(df, x="Pclass", color="SurvivalStatus", barmode="group",
                            title="Распределение выживших по классу билета",
                            labels={'SurvivalStatus': 'Статус выживания'},
                            color_discrete_map={'Не выжил': '#EF553B', 'Выжил': '#636EFA'})
        st.plotly_chart(fig_pclass, use_container_width=True)

    with col2:
        # Просто показываем гистограмму по Sex
        fig_sex = px.histogram(df, x="Sex", color="SurvivalStatus", barmode="group",
                            title="Распределение выживших по полу",
                            labels={'SurvivalStatus': 'Статус выживания'},
                            color_discrete_map={'Не выжил': '#EF553B', 'Выжил': '#636EFA'})
        st.plotly_chart(fig_sex, use_container_width=True)

    st.write("### 📈 Распределение по возрасту")
    fig_age = px.histogram(df, x="Age", color="SurvivalStatus", barmode="overlay",
                           title="Распределение выживших по возрасту",
                           labels={'SurvivalStatus': 'Статус выживания'},
                           color_discrete_map={'Не выжил': '#EF553B', 'Выжил': '#636EFA'})
    st.plotly_chart(fig_age, use_container_width=True)

# --- Подготовка данных для моделей (для вкладок 2 и 3 - УПРОЩЕН) ---
# Исключаем FamilySize, IsAlone, Title
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Encoder теперь кодирует только 'Sex' и 'Embarked'
encoder = ce.TargetEncoder(cols=['Sex', 'Embarked'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

# --- ВКЛАДКА 2: ОБУЧЕНИЕ И НАСТРОЙКА МОДЕЛЕЙ ---
with tab2:
    st.header("Выбор и настройка модели машинного обучения")
    model_choice = st.selectbox(
        "Выберите модель для обучения:",
        ("Decision Tree", "K-Nearest Neighbors", "Logistic Regression", "Random Forest")
    )

    params = {}
    model_trained = None
    if model_choice == "Decision Tree":
        params['max_depth'] = st.slider("Максимальная глубина дерева (max_depth)", 2, 20, 5, 1)
        params['min_samples_leaf'] = st.slider("Мин. число объектов в листе (min_samples_leaf)", 1, 50, 5, 1)
        model_trained = DecisionTreeClassifier(random_state=42, **params)
    elif model_choice == "K-Nearest Neighbors":
        params['n_neighbors'] = st.slider("Число соседей (n_neighbors)", 1, 20, 5, 1)
        model_trained = KNeighborsClassifier(**params)
    elif model_choice == "Logistic Regression":
        params['C'] = st.slider("Сила регуляризации (C)", 0.01, 10.0, 1.0, 0.01)
        model_trained = LogisticRegression(random_state=42, max_iter=1000, **params)
    elif model_choice == "Random Forest":
        params['n_estimators'] = st.slider("Количество деревьев (n_estimators)", 50, 500, 100, 10)
        params['max_depth'] = st.slider("Максимальная глубина дерева (max_depth)", 2, 20, 7, 1)
        model_trained = RandomForestClassifier(random_state=42, **params)

    if st.button("🚀 Обучить и оценить модель", use_container_width=True):
        if model_trained is not None:
            model_trained.fit(X_train_encoded, y_train)
            y_pred_train = model_trained.predict(X_train_encoded)
            y_pred_test = model_trained.predict(X_test_encoded)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)

            st.write("### Результаты оценки:")
            col1, col2 = st.columns(2)
            col1.metric("Точность на обучающей выборке", f"{acc_train:.2%}")
            col2.metric("Точность на тестовой выборке", f"{acc_test:.2%}")

            cm = confusion_matrix(y_test, y_pred_test)
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                labels=dict(x="Предсказанный класс", y="Истинный класс", color="Количество"),
                                x=['Не выжил', 'Выжил'], y=['Не выжил', 'Выжил'],
                                color_continuous_scale='Blues',
                                title="Матрица ошибок")
            st.plotly_chart(fig_cm, use_container_width=True)

            st.session_state['model'] = model_trained
            st.session_state['encoder'] = encoder
            st.session_state['X_train_encoded_columns'] = X_train_encoded.columns.tolist()
            st.success("Модель успешно обучена и готова для прогнозирования!")
        else:
            st.warning("Пожалуйста, выберите модель для обучения.")

# --- ВКЛАДКА 3: СДЕЛАТЬ ПРОГНОЗ (УПРОЩЕН) ---
with tab3:
    st.header("Прогноз выживаемости пассажира")
    if 'model' not in st.session_state or 'encoder' not in st.session_state:
        st.warning("Сначала обучите модель на вкладке 'Обучение и настройка моделей'!")
    else:
        model_to_predict = st.session_state['model']
        encoder_to_predict = st.session_state['encoder']
        X_train_cols = st.session_state['X_train_encoded_columns']

        st.info("Введите параметры пассажира, чтобы получить прогноз.")
        col1, col2 = st.columns(2)
        with col1:
            pclass_input = st.selectbox("Класс билета", sorted(df['Pclass'].unique()), key='pclass_simple')
            sex_input = st.selectbox("Пол", df['Sex'].unique(), key='sex_simple')
            age_input = st.slider("Возраст", 0, 100, 30, key='age_simple')
            fare_input = st.slider("Стоимость билета", 0.0, float(df['Fare'].max()), 32.0, key='fare_simple')

        # Убрали SibSp, Parch, Embarked и Title из прямого ввода пользователя
        # Эти значения будут заданы по умолчанию в DataFrame для предсказания

        if st.button("Получить прогноз", use_container_width=True, type="primary"):
            # Создаем DataFrame с упрощенным вводом
            user_input = pd.DataFrame([{
                'Pclass': pclass_input,
                'Sex': sex_input,
                'Age': age_input,
                'SibSp': 0,  # Значение по умолчанию
                'Parch': 0,  # Значение по умолчанию
                'Fare': fare_input,
                'Embarked': 'S' # Значение по умолчанию (самый частый порт)
            }])

            user_encoded = encoder_to_predict.transform(user_input)
            user_encoded = user_encoded[X_train_cols]

            prediction = model_to_predict.predict(user_encoded)[0]
            probability = model_to_predict.predict_proba(user_encoded)[0]

            result_col1, result_col2 = st.columns([1, 2])
            with result_col1:
                if prediction == 1:
                    st.metric(label="Прогноз", value="Выжил", delta="Высокие шансы")
                    st.image("https://em-content.zobj.net/source/microsoft-teams/363/lifebuoy_1f6df.png", width=150)
                else:
                    st.metric(label="Прогноз", value="Не выжил", delta="Низкие шансы", delta_color="inverse")
                    st.image("https://em-content.zobj.net/source/microsoft-teams/363/skull-and-crossbones_2620-fe0f.png", width=150)

            with result_col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability[1] * 100,
                    title = {'text': "Вероятность выжить (%)"},
                    gauge = {'axis': {'range': [None, 100]},
                             'bar': {'color': "#636EFA"},
                             'steps' : [
                                 {'range': [0, 50], 'color': "#F0F2F6"},
                                 {'range': [50, 100], 'color': "#D6EAF8"}],
                             }))
                st.plotly_chart(fig_gauge, use_container_width=True)
                

# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix 
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression 
# from sklearn.ensemble import RandomForestClassifier 
# import category_encoders as ce
# import plotly.express as px
# import plotly.graph_objects as go


# st.set_page_config(page_title="🚢 Прогноз выживания на Титанике", layout="wide")
# st.title('🚢 Прогноз выживания на Титанике - Обучение и предсказание')
# st.write('## Работа с датасетом Титаника')

# df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# # Заполняем пропуски в 'Age' медианным значением
# df['Age'].fillna(df['Age'].median(), inplace=True)
# # Заполняем пропуски в 'Embarked' самым частым значением (модой)
# df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# df['SurvivalStatus'] = df['Survived'].map({0: 'Не выжил', 1: 'Выжил'})

# df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# df['IsAlone'] = (df['FamilySize'] == 1).astype(int) #  int 
# df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
# # Объединение редких титулов 
# rare_titles = df['Title'].value_counts() < 10
# df['Title'] = df['Title'].apply(lambda x: 'Rare' if rare_titles[x] else x)

# st.subheader("🔍 Случайные 10 строк")
# st.dataframe(df.sample(10), use_container_width=True)

# st.subheader("📊 Визуализация данных")
# col1, col2 = st.columns(2)

# with col1:
#     # Распределение выживших по классу билета
#     fig1 = px.histogram(df, x="Pclass", color="SurvivalStatus", barmode="group",
#                         title="Распределение выживших по классу билета",
#                         labels={'Pclass': 'Класс билета', 'SurvivalStatus': 'Статус выживания'})
#     st.plotly_chart(fig1, use_container_width=True)

# with col2:
#     # Зависимость выживаемости от возраста и пола
#     fig2 = px.scatter(df, x="Age", y="Fare", color="SurvivalStatus",
#                       title="Стоимость билета vs Возраст",
#                       labels={'Age': 'Возраст', 'Fare': 'Стоимость билета', 'SurvivalStatus': 'Статус выживания'})
#     st.plotly_chart(fig2, use_container_width=True)

# features_for_initial_comparison = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# X_initial_comparison = df[features_for_initial_comparison]
# y_initial_comparison = df['Survived']

# X_train_initial_comparison, X_test_initial_comparison, y_train_initial_comparison, y_test_initial_comparison = train_test_split(X_initial_comparison, y_initial_comparison, test_size=0.3, random_state=42, stratify=y_initial_comparison)

# # ---  категориальные признаки для верхней секции ---
# encoder_initial_comparison = ce.TargetEncoder(cols=['Sex', 'Embarked'])
# X_train_encoded_initial_comparison = encoder_initial_comparison.fit_transform(X_train_initial_comparison, y_train_initial_comparison)
# X_test_encoded_initial_comparison = encoder_initial_comparison.transform(X_test_initial_comparison)

# # --- Обучение и оценка моделей ---
# models = {
#     'Decision Tree': DecisionTreeClassifier(random_state=42),
#     'KNN': KNeighborsClassifier()
# }

# results = []
# for name, model in models.items():
#     model.fit(X_train_encoded_initial_comparison, y_train_initial_comparison)
#     acc_train = accuracy_score(y_train_initial_comparison, model.predict(X_train_encoded_initial_comparison))
#     acc_test = accuracy_score(y_test_initial_comparison, model.predict(X_test_encoded_initial_comparison))
#     results.append({
#         'Model': name,
#         'Train Accuracy': round(acc_train, 2),
#         'Test Accuracy': round(acc_test, 2)
#     })

# st.write("### 📊 Сравнение моделей по точности")
# st.table(pd.DataFrame(results))

# # --- Виджет в боковой панели ---
# st.sidebar.header("🔮 Предсказание по параметрам")

# # Виджеты для ввода данных пользователем
# pclass_input_sb = st.sidebar.selectbox("Класс билета", sorted(df['Pclass'].unique()), key='sb_pclass')
# sex_input_sb = st.sidebar.selectbox("Пол", df['Sex'].unique(), key='sb_sex')
# age_input_sb = st.sidebar.slider("Возраст", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()), key='sb_age')
# sibsp_input_sb = st.sidebar.slider("Кол-во братьев/сестер/супругов на борту", int(df['SibSp'].min()), int(df['SibSp'].max()), int(df['SibSp'].mode()[0]), key='sb_sibsp')
# parch_input_sb = st.sidebar.slider("Кол-во родителей/детей на борту", int(df['Parch'].min()), int(df['Parch'].max()), int(df['Parch'].mode()[0]), key='sb_parch')
# fare_input_sb = st.sidebar.slider("Стоимость билета", float(df['Fare'].min()), float(df['Fare'].max()), float(df['Fare'].median()), key='sb_fare')
# embarked_input_sb = st.sidebar.selectbox("Порт посадки", df['Embarked'].unique(), key='sb_embarked')

# user_input_sb = pd.DataFrame([{
#     'Pclass': pclass_input_sb,
#     'Sex': sex_input_sb,
#     'Age': age_input_sb,
#     'SibSp': sibsp_input_sb,
#     'Parch': parch_input_sb,
#     'Fare': fare_input_sb,
#     'Embarked': embarked_input_sb
# }])

# user_encoded_sb = encoder_initial_comparison.transform(user_input_sb)
# user_encoded_sb = user_encoded_sb[X_train_encoded_initial_comparison.columns]


# st.sidebar.subheader("📈 Результаты предсказания")
# for name, model in models.items():
#     pred_val = model.predict(user_encoded_sb)[0]
#     pred_text = "Выжил" if pred_val == 1 else "Не выжил"
#     proba = model.predict_proba(user_encoded_sb)[0]

#     st.sidebar.markdown(f"**{name}: {pred_text}**")
#     proba_df = pd.DataFrame({
#         'Статус': ['Не выжил', 'Выжил'],
#         'Вероятность': proba
#     })
#     st.sidebar.dataframe(proba_df.set_index("Статус"), use_container_width=True)


# # --- Создание вкладок ---
# tab1, tab2, tab3 = st.tabs(["📊 Анализ данных", "🤖 Обучение и настройка моделей", "🔮 Сделать прогноз"])

# # --- ВКЛАДКА 1: АНАЛИЗ ДАННЫХ ---
# with tab1:
#     st.header("Обзор и визуализация данных")
#     st.write("### 📋 Данные о пассажирах")
#     st.dataframe(df.head(10), use_container_width=True)

#     st.write("### 📈 Интерактивные графики")
#     col1, col2 = st.columns(2)
#     with col1:
#         feature_to_plot = st.selectbox(
#             "Выберите признак для анализа распределения выживших:",
#             ('Pclass', 'Sex', 'Embarked', 'FamilySize', 'Title') 
#         )
#         fig1 = px.histogram(df, x=feature_to_plot, color="SurvivalStatus", barmode="group",
#                             title=f"Распределение выживших по признаку '{feature_to_plot}'",
#                             labels={'SurvivalStatus': 'Статус выживания'},
#                             color_discrete_map={'Не выжил': '#EF553B', 'Выжил': '#636EFA'})
#         st.plotly_chart(fig1, use_container_width=True)

#     with col2:
#         # Violin plot для анализа возраста
#         fig2 = px.violin(df, x="Sex", y="Age", color="SurvivalStatus", box=True, points="all",
#                          title="Распределение возраста по полу и статусу выживания",
#                          labels={'Sex': 'Пол', 'Age': 'Возраст', 'SurvivalStatus': 'Статус выживания'},
#                          color_discrete_map={'Не выжил': '#EF553B', 'Выжил': '#636EFA'})
#         st.plotly_chart(fig2, use_container_width=True)

#     st.write("### ☀️ Иерархия выживания")
#     fig3 = px.sunburst(df, path=['Pclass', 'Sex', 'SurvivalStatus'],
#                                title="Иерархическое распределение выживших по классу и полу",
#                                color_discrete_map={'(?)':'gold', 'Не выжил': '#EF553B', 'Выжил': '#636EFA'})
#     st.plotly_chart(fig3, use_container_width=True)


# features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
# X = df[features] 
# y = df['Survived']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# # Encoder для всех признаков, включая новые, для использования в табах
# encoder = ce.TargetEncoder(cols=['Sex', 'Embarked', 'Title', 'IsAlone'])
# X_train_encoded = encoder.fit_transform(X_train, y_train)
# X_test_encoded = encoder.transform(X_test)

# # --- ВКЛАДКА 2: ОБУЧЕНИЕ И НАСТРОЙКА МОДЕЛЕЙ ---
# with tab2:
#     st.header("Выбор и настройка модели машинного обучения")
#     model_choice = st.selectbox(
#         "Выберите модель для обучения:",
#         ("Decision Tree", "K-Nearest Neighbors", "Logistic Regression", "Random Forest")
#     )

#     params = {}
#     model_trained = None
#     if model_choice == "Decision Tree":
#         params['max_depth'] = st.slider("Максимальная глубина дерева (max_depth)", 2, 20, 5, 1)
#         params['min_samples_leaf'] = st.slider("Мин. число объектов в листе (min_samples_leaf)", 1, 50, 5, 1)
#         model_trained = DecisionTreeClassifier(random_state=42, **params)
#     elif model_choice == "K-Nearest Neighbors":
#         params['n_neighbors'] = st.slider("Число соседей (n_neighbors)", 1, 20, 5, 1)
#         model_trained = KNeighborsClassifier(**params)
#     elif model_choice == "Logistic Regression":
#         params['C'] = st.slider("Сила регуляризации (C)", 0.01, 10.0, 1.0, 0.01)
#         model_trained = LogisticRegression(random_state=42, max_iter=1000, **params)
#     elif model_choice == "Random Forest":
#         params['n_estimators'] = st.slider("Количество деревьев (n_estimators)", 50, 500, 100, 10)
#         params['max_depth'] = st.slider("Максимальная глубина дерева (max_depth)", 2, 20, 7, 1)
#         model_trained = RandomForestClassifier(random_state=42, **params)

#     if st.button("🚀 Обучить и оценить модель", use_container_width=True):
#         if model_trained is not None:
#             model_trained.fit(X_train_encoded, y_train)
#             y_pred_train = model_trained.predict(X_train_encoded)
#             y_pred_test = model_trained.predict(X_test_encoded)
#             acc_train = accuracy_score(y_train, y_pred_train)
#             acc_test = accuracy_score(y_test, y_pred_test)

#             st.write("### Результаты оценки:")
#             col1, col2 = st.columns(2)
#             col1.metric("Точность на обучающей выборке", f"{acc_train:.2%}")
#             col2.metric("Точность на тестовой выборке", f"{acc_test:.2%}")

#             cm = confusion_matrix(y_test, y_pred_test)
#             fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
#                                 labels=dict(x="Предсказанный класс", y="Истинный класс", color="Количество"),
#                                 x=['Не выжил', 'Выжил'], y=['Не выжил', 'Выжил'],
#                                 color_continuous_scale='Blues',
#                                 title="Матрица ошибок")
#             st.plotly_chart(fig_cm, use_container_width=True)

#             st.session_state['model'] = model_trained
#             st.session_state['encoder'] = encoder 
#             st.session_state['X_train_encoded_columns'] = X_train_encoded.columns.tolist() 
#             st.success("Модель успешно обучена и готова для прогнозирования!")
#         else:
#             st.warning("Пожалуйста, выберите модель для обучения.")


# # --- ВКЛАДКА 3: СДЕЛАТЬ ПРОГНОЗ ---
# with tab3:
#     st.header("Прогноз выживаемости пассажира")
#     if 'model' not in st.session_state or 'encoder' not in st.session_state:
#         st.warning("Сначала обучите модель на вкладке 'Обучение и настройка моделей'!")
#     else:
#         # Получаем модель и энкодер из session_state
#         model_to_predict = st.session_state['model']
#         encoder_to_predict = st.session_state['encoder']
#         X_train_cols = st.session_state['X_train_encoded_columns']


#         st.info("Введите параметры пассажира, чтобы получить прогноз.")
#         col1, col2 = st.columns(2)
#         with col1:
#             pclass_input = st.selectbox("Класс билета", sorted(df['Pclass'].unique()), key='pclass')
#             sex_input = st.selectbox("Пол", df['Sex'].unique(), key='sex')
#             age_input = st.slider("Возраст", 0, 100, 30, key='age')
#             fare_input = st.slider("Стоимость билета", 0.0, float(df['Fare'].max()), 32.0, key='fare')

#         with col2:
#             sibsp_input = st.number_input("Кол-во братьев/сестер/супругов", min_value=0, max_value=10, value=0, key='sibsp')
#             parch_input = st.number_input("Кол-во родителей/детей", min_value=0, max_value=10, value=0, key='parch')
#             embarked_input = st.selectbox("Порт посадки", df['Embarked'].unique(), key='embarked')
#             # Выбор титула из уже обработанных уникальных значений
#             title_input = st.selectbox("Титул", df['Title'].unique(), key='title')


#         if st.button("Получить прогноз", use_container_width=True, type="primary"):
#             # Вычисляем FamilySize и IsAlone для пользовательского ввода
#             family_size = sibsp_input + parch_input + 1
#             is_alone = int(family_size == 1) # Преобразуем True/False  в 1/0

#             user_input = pd.DataFrame([{
#                 'Pclass': pclass_input,
#                 'Sex': sex_input,
#                 'Age': age_input,
#                 'SibSp': sibsp_input,
#                 'Parch': parch_input,
#                 'Fare': fare_input,
#                 'Embarked': embarked_input,
#                 'FamilySize': family_size, 
#                 'IsAlone': is_alone,      
#                 'Title': title_input      
#             }])

#             user_encoded = encoder_to_predict.transform(user_input)
#             # Убедимся, что порядок столбцов совпадает с обучающими данными
#             user_encoded = user_encoded[X_train_cols]

#             # Делаем предсказание
#             prediction = model_to_predict.predict(user_encoded)[0]
#             probability = model_to_predict.predict_proba(user_encoded)[0]

#             # Отображаем результат
#             result_col1, result_col2 = st.columns([1, 2])
#             with result_col1:
#                 if prediction == 1:
#                     st.metric(label="Прогноз", value="Выжил", delta="Высокие шансы")
#                     st.image("https://em-content.zobj.net/source/microsoft-teams/363/lifebuoy_1f6df.png", width=150)
#                 else:
#                     st.metric(label="Прогноз", value="Не выжил", delta="Низкие шансы", delta_color="inverse")
#                     st.image("https://em-content.zobj.net/source/microsoft-teams/363/skull-and-crossbones_2620-fe0f.png", width=150)

#             with result_col2:
#                 # Gauge chart для вероятности
#                 fig_gauge = go.Figure(go.Indicator(
#                     mode = "gauge+number",
#                     value = probability[1] * 100,
#                     title = {'text': "Вероятность выжить (%)"},
#                     gauge = {'axis': {'range': [None, 100]},
#                              'bar': {'color': "#636EFA"},
#                              'steps' : [
#                                  {'range': [0, 50], 'color': "#F0F2F6"},
#                                  {'range': [50, 100], 'color': "#D6EAF8"}],
#                              }))
#                 st.plotly_chart(fig_gauge, use_container_width=True)

