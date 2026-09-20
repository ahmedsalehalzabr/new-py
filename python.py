import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# 1. إنشاء جدول البيانات
data = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Title": [
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Toy Story 4",
        "Parasite",
        "Oppenheimer",
        "Spider-Man: Spider-Verse",
        "Pulp Fiction",
        "Dune: Part Two",
        "Spirited Away",
    ],
    "Has_Oscar": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Is_Streaming": [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    "Release_Year": [2010, 2014, 2008, 2019, 2019, 2023, 2018, 1994, 2024, 2001],
    "IMDb_Rating": [8.8, 8.7, 9.0, 7.7, 8.5, 8.9, 8.4, 8.9, 8.6, 8.6],
    "Duration_Min": [148, 169, 152, 100, 132, 180, 117, 154, 166, 125],
    "Genre": [
        "Sci-Fi",
        "Sci-Fi",
        "Action",
        "Animation",
        "Drama",
        "Drama",
        "Animation",
        "Crime",
        "Sci-Fi",
        "Animation",
    ],
}

df = pd.DataFrame(data)

# 2. معالجة وتطبيع البيانات الرقمية (Feature Scaling)
scaler = MinMaxScaler()
numeric_features = ["Release_Year", "IMDb_Rating", "Duration_Min"]
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

# 3. اختيار خيارات المتجه للتوصية
features = ["Has_Oscar", "Is_Streaming"] + numeric_features
feature_matrix = df_scaled[features].values

# 4. حساب مصفوفة التشابه (Cosine Similarity Matrix)
similarity_matrix = cosine_similarity(feature_matrix)


# 5. دالة التوصية بناءً على الفيلم المختار 
def recommend_movie(movie_title, top_n=3):
    if movie_title not in df["Title"].values:
        return "الفيلم غير موجود بالقائمة!"

    idx = df[df["Title"] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\n🎬 الأفلام الموصى بها لمشاهدي فيلم [{movie_title}]:")
    print("-" * 50)
    count = 0
    for i, score in scores:
        if i != idx:  # تجاهل الفيلم نفسه
            rec_title = df.loc[i, "Title"]
            genre = df.loc[i, "Genre"]
            rating = df.loc[i, "IMDb_Rating"]
            print(
                f"{count+1}. {rec_title} | النوع: {genre} | التقييم: {rating} | نسبة التشابه: {score*100:.1f}%"
            )
            count += 1
            if count == top_n:
                break


# تجربة الخوارزمية
recommend_movie("Inception", top_n=3)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# 1. Создание таблицы данных
data = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Название": [
        "Начало",
        "Интерстеллар",
        "Тёмный рыцарь",
        "История игрушек 4",
        "Паразиты",
        "Оппенгеймер",
        "Человек-паук: Через вселенные",
        "Криминальное чтиво",
        "Дюна: Часть вторая",
        "Унесённые призраками",
    ],
    "Оскар": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Доступен_для_просмотра": [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    "Год_выпуска": [2010, 2014, 2008, 2019, 2019, 2023, 2018, 1994, 2024, 2001],
    "Рейтинг_IMDb": [8.8, 8.7, 9.0, 7.7, 8.5, 8.9, 8.4, 8.9, 8.6, 8.6],
    "Продолжительность_мин": [148, 169, 152, 100, 132, 180, 117, 154, 166, 125],
    "Жанр": [
        "Научная фантастика",
        "Научная фантастика",
        "Боевик",
        "Мультфильм",
        "Драма",
        "Драма",
        "Мультфильм",
        "Криминал",
        "Научная фантастика",
        "Мультфильм",
    ],
}

df = pd.DataFrame(data)

# 2. Обработка и нормализация числовых данных
scaler = MinMaxScaler()

numeric_features = [
    "Год_выпуска",
    "Рейтинг_IMDb",
    "Продолжительность_мин"
]

df_scaled = df.copy()

df_scaled[numeric_features] = scaler.fit_transform(
    df[numeric_features]
)

# 3. Выбор характеристик для рекомендации
features = [
    "Оскар",
    "Доступен_для_просмотра"
] + numeric_features

feature_matrix = df_scaled[features].values

# 4. Расчёт матрицы косинусного сходства
similarity_matrix = cosine_similarity(feature_matrix)


# 5. Функция рекомендации фильмов
def рекомендовать_фильмы(название_фильма, количество=3):

    if название_фильма not in df["Название"].values:
        return "Фильм не найден в списке!"

    индекс = df[df["Название"] == название_фильма].index[0]

    оценки = list(
        enumerate(similarity_matrix[индекс])
    )

    оценки = sorted(
        оценки,
        key=lambda x: x[1],
        reverse=True
    )

    print(
        f"\n🎬 Рекомендуемые фильмы для зрителей фильма "
        f"[{название_фильма}]:"
    )

    print("-" * 60)

    счетчик = 0

    for i, сходство in оценки:

        if i != индекс:

            название = df.loc[i, "Название"]
            жанр = df.loc[i, "Жанр"]
            рейтинг = df.loc[i, "Рейтинг_IMDb"]

            print(
                f"{счетчик + 1}. {название} | "
                f"Жанр: {жанр} | "
                f"Рейтинг: {рейтинг} | "
                f"Сходство: {сходство * 100:.1f}%"
            )

            счетчик += 1

            if счетчик == количество:
                break


# 6. Проверка алгоритма
рекомендовать_фильмы("Начало", количество=3)


