import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 🎬 СИСТЕМА РЕКОМЕНДАЦИЙ ФИЛЬМОВ КИНОТЕАТРА
# ============================================================

# -----------------------------
# 1. НАБОР ДАННЫХ О ФИЛЬМАХ
# -----------------------------

data = {
    "ID": list(range(1, 21)),

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
        "Матрица",
        "Бойцовский клуб",
        "Форрест Гамп",
        "Гладиатор",
        "Властелин колец: Братство Кольца",
        "Мстители: Финал",
        "Джокер",
        "Отель Гранд Будапешт",
        "Бегущий по лезвию 2049",
        "Душа"
    ],

    "Оскар": [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 1, 1, 1, 0, 1, 1, 1, 1
    ],

    "В_Прокате": [
        1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ],

    "Год_Выпуска": [
        2010, 2014, 2008, 2019, 2019,
        2023, 2018, 1994, 2024, 2001,
        1999, 1999, 1994, 2000, 2001,
        2019, 2019, 2014, 2017, 2020
    ],

    "Рейтинг_IMDb": [
        8.8, 8.7, 9.0, 7.7, 8.5,
        8.9, 8.4, 8.9, 8.6, 8.6,
        8.7, 8.8, 8.8, 8.5, 8.9,
        8.4, 8.4, 8.1, 8.0, 8.0
    ],

    "Длительность_Мин": [
        148, 169, 152, 100, 132,
        180, 117, 154, 166, 125,
        136, 139, 142, 155, 178,
        181, 122, 99, 164, 100
    ],

    "Жанр": [
        "Фантастика / Боевик",
        "Фантастика / Драма",
        "Боевик / Криминал",
        "Анимация / Семейный",
        "Драма / Триллер",
        "Биография / Драма",
        "Анимация / Боевик",
        "Криминал / Драма",
        "Фантастика / Приключения",
        "Анимация / Фэнтези",
        "Фантастика / Боевик",
        "Драма / Триллер",
        "Драма / Романтика",
        "Боевик / Драма",
        "Фэнтези / Приключения",
        "Боевик / Фантастика",
        "Драма / Триллер",
        "Комедия / Драма",
        "Фантастика / Драма",
        "Анимация / Фэнтези"
    ],

    "Режиссёр": [
        "Кристофер Нолан",
        "Кристофер Нолан",
        "Кристофер Нолан",
        "Джош Кули",
        "Пон Джун Хо",
        "Кристофер Нолан",
        "Боб Персикетти",
        "Квентин Тарантино",
        "Дени Вильнёв",
        "Хаяо Миядзаки",
        "Братья Вачовски",
        "Дэвид Финчер",
        "Роберт Земекис",
        "Ридли Скотт",
        "Питер Джексон",
        "Братья Руссо",
        "Тодд Филлипс",
        "Уэс Андерсон",
        "Дени Вильнёв",
        "Пит Доктер"
    ]
}

df = pd.DataFrame(data)


# ============================================================
# 2. НОРМАЛИЗАЦИЯ ДАННЫХ
# ============================================================

numeric_features = [
    "Год_Выпуска",
    "Рейтинг_IMDb",
    "Длительность_Мин"
]

scaler = MinMaxScaler()

df_scaled = df.copy()

df_scaled[numeric_features] = scaler.fit_transform(
    df[numeric_features]
)


# ============================================================
# 3. МАТРИЦА ПРИЗНАКОВ
# ============================================================

features = [
    "Оскар",
    "В_Прокате",
    "Год_Выпуска",
    "Рейтинг_IMDb",
    "Длительность_Мин"
]

feature_matrix = df_scaled[features].values


# ============================================================
# 4. КОСИНУСНОЕ СХОДСТВО
# ============================================================

similarity_matrix = cosine_similarity(feature_matrix)


# ============================================================
# 5. ФУНКЦИЯ РЕКОМЕНДАЦИЙ
# ============================================================

def get_recommendations(movie_title, top_n=5):

    if movie_title not in df["Название"].values:
        return []

    index = df[df["Название"] == movie_title].index[0]

    scores = list(
        enumerate(similarity_matrix[index])
    )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for movie_index, score in scores:

        if movie_index == index:
            continue

        recommendations.append({
            "Название": df.loc[movie_index, "Название"],
            "Жанр": df.loc[movie_index, "Жанр"],
            "Режиссёр": df.loc[movie_index, "Режиссёр"],
            "Рейтинг": df.loc[movie_index, "Рейтинг_IMDb"],
            "Год": df.loc[movie_index, "Год_Выпуска"],
            "Длительность": df.loc[movie_index, "Длительность_Мин"],
            "Сходство": score * 100
        })

        if len(recommendations) >= top_n:
            break

    return recommendations


# ============================================================
# 6. ГЛАВНОЕ ОКНО
# ============================================================

root = tk.Tk()

root.title(
    "🎬 Система рекомендаций фильмов кинотеатра"
)

root.geometry("1350x800")

root.minsize(1100, 700)

# Цветовая схема
BG_COLOR = "#1a1a2e"
FG_COLOR = "#eaeaea"
ACCENT_COLOR = "#e94560"
SECONDARY_BG = "#16213e"
TEXT_BG = "#0f3460"

root.configure(bg=BG_COLOR)


# ============================================================
# 7. ЗАГОЛОВОК
# ============================================================

title = tk.Label(
    root,
    text="🎬 СИСТЕМА РЕКОМЕНДАЦИЙ ФИЛЬМОВ КИНОТЕАТРА",
    font=("Arial", 24, "bold"),
    bg=BG_COLOR,
    fg=ACCENT_COLOR
)

title.pack(pady=15)


subtitle = tk.Label(
    root,
    text="Рекомендательные системы и таксономия данных",
    font=("Arial", 13),
    bg=BG_COLOR,
    fg=FG_COLOR
)

subtitle.pack()


# ============================================================
# 8. ВКЛАДКИ (NOTEBOOK)
# ============================================================

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TNotebook",
    background=BG_COLOR,
    borderwidth=0
)

style.configure(
    "TNotebook.Tab",
    background=SECONDARY_BG,
    foreground=FG_COLOR,
    padding=[15, 8],
    font=("Arial", 11, "bold")
)

style.map(
    "TNotebook.Tab",
    background=[("selected", ACCENT_COLOR)],
    foreground=[("selected", "white")]
)

notebook = ttk.Notebook(root)

notebook.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=15
)


# ============================================================
# ВКЛАДКА 1 - НАБОР ДАННЫХ
# ============================================================

dataset_tab = ttk.Frame(notebook)

notebook.add(
    dataset_tab,
    text="📊 Набор данных"
)


columns = [
    "ID",
    "Название",
    "Оскар",
    "В_Прокате",
    "Год",
    "IMDb",
    "Длительность",
    "Жанр",
    "Режиссёр"
]


tree = ttk.Treeview(
    dataset_tab,
    columns=columns,
    show="headings"
)

for column in columns:

    tree.heading(
        column,
        text=column
    )

    tree.column(
        column,
        width=130
    )


for _, row in df.iterrows():

    tree.insert(
        "",
        "end",
        values=(
            row["ID"],
            row["Название"],
            row["Оскар"],
            row["В_Прокате"],
            row["Год_Выпуска"],
            row["Рейтинг_IMDb"],
            row["Длительность_Мин"],
            row["Жанр"],
            row["Режиссёр"]
        )
    )


scrollbar = ttk.Scrollbar(
    dataset_tab,
    orient="vertical",
    command=tree.yview
)

tree.configure(
    yscrollcommand=scrollbar.set
)

tree.pack(
    side="left",
    fill="both",
    expand=True
)

scrollbar.pack(
    side="right",
    fill="y"
)


# ============================================================
# ВКЛАДКА 2 - РЕКОМЕНДАЦИИ
# ============================================================

recommend_tab = ttk.Frame(notebook)

notebook.add(
    recommend_tab,
    text="🎬 Рекомендации"
)


label = tk.Label(
    recommend_tab,
    text="Выберите фильм:",
    font=("Arial", 15, "bold"),
    bg=BG_COLOR,
    fg=FG_COLOR
)

label.pack(pady=15)


movie_var = tk.StringVar()

movie_box = ttk.Combobox(
    recommend_tab,
    textvariable=movie_var,
    values=list(df["Название"]),
    width=50,
    state="readonly",
    font=("Arial", 12)
)

movie_box.pack(pady=10)

movie_box.current(0)


result_text = tk.Text(
    recommend_tab,
    font=("Consolas", 12),
    height=20,
    width=100,
    bg=TEXT_BG,
    fg=FG_COLOR,
    insertbackground=FG_COLOR
)

result_text.pack(
    padx=20,
    pady=20,
    fill="both",
    expand=True
)


def show_recommendations():

    movie = movie_var.get()

    recommendations = get_recommendations(
        movie,
        5
    )

    result_text.delete(
        "1.0",
        tk.END
    )

    result_text.insert(
        tk.END,
        f"🎬 РЕКОМЕНДАЦИИ ДЛЯ: {movie}\n"
    )

    result_text.insert(
        tk.END,
        "=" * 90 + "\n\n"
    )

    for i, rec in enumerate(
        recommendations,
        start=1
    ):

        result_text.insert(
            tk.END,
            f"{i}. {rec['Название']} ({rec['Год']})\n"
        )

        result_text.insert(
            tk.END,
            f"   Жанр: {rec['Жанр']}\n"
        )

        result_text.insert(
            tk.END,
            f"   Режиссёр: {rec['Режиссёр']}\n"
        )

        result_text.insert(
            tk.END,
            f"   Рейтинг IMDb: {rec['Рейтинг']}\n"
        )

        result_text.insert(
            tk.END,
            f"   Длительность: {rec['Длительность']} мин\n"
        )

        result_text.insert(
            tk.END,
            f"   Сходство: {rec['Сходство']:.2f}%\n\n"
        )


button = ttk.Button(
    recommend_tab,
    text="🔎 Сгенерировать рекомендации",
    command=show_recommendations
)

button.pack(pady=10)


# ============================================================
# ВКЛАДКА 3 - МАТРИЦА СХОДСТВА
# ============================================================

similarity_tab = ttk.Frame(notebook)

notebook.add(
    similarity_tab,
    text="🔢 Матрица сходства"
)


similarity_tree = ttk.Treeview(
    similarity_tab,
    columns=list(df["Название"]),
    show="headings"
)

for title_name in df["Название"]:

    similarity_tree.heading(
        title_name,
        text=title_name[:12]
    )

    similarity_tree.column(
        title_name,
        width=100
    )


for i, title_name in enumerate(df["Название"]):

    values = []

    for j in range(len(df)):

        values.append(
            f"{similarity_matrix[i][j]:.2f}"
        )

    similarity_tree.insert(
        "",
        "end",
        text=title_name,
        values=values
    )


similarity_scrollbar = ttk.Scrollbar(
    similarity_tab,
    orient="vertical",
    command=similarity_tree.yview
)

similarity_tree.configure(
    yscrollcommand=similarity_scrollbar.set
)

similarity_tree.pack(
    side="left",
    fill="both",
    expand=True
)

similarity_scrollbar.pack(
    side="right",
    fill="y"
)


# ============================================================
# ВКЛАДКА 4 - ТАКСОНОМИЯ
# ============================================================

taxonomy_tab = ttk.Frame(notebook)

notebook.add(
    taxonomy_tab,
    text="🌳 Таксономия"
)


taxonomy_text = tk.Text(
    taxonomy_tab,
    font=("Consolas", 14),
    padx=30,
    pady=30,
    bg=TEXT_BG,
    fg=FG_COLOR,
    insertbackground=FG_COLOR
)

taxonomy_text.pack(
    fill="both",
    expand=True
)


taxonomy = """
                    🎬 ФИЛЬМЫ КИНОТЕАТРА
                           │
             ┌─────────────┴─────────────┐
             │                           │
       🎥 ИГРОВОЕ КИНО              🎨 АНИМАЦИЯ
             │                           │
      ┌──────┼──────────┐          ┌─────┼──────┐
      │      │          │          │     │      │
  Фантастика Драма   Криминал   Семейный Боевик Фэнтези
      │      │          │          │     │      │
      │      │          │          │     │      │
  Начало   Паразиты  Криминальное История  Человек- Унесённые
  Интер-   Оппен-    чтиво        игрушек  паук:   призраками
  стеллар  геймер    Бойцовский   4        Через   Душа
  Матрица  Форрест   клуб                 вселенные
  Дюна     Гамп      Джокер
  Тёмный   Гладиатор
  рыцарь
"""


taxonomy_text.insert(
    tk.END,
    taxonomy
)

taxonomy_text.config(
    state="disabled"
)


# ============================================================
# ВКЛАДКА 5 - СТАТИСТИКА
# ============================================================

stats_tab = ttk.Frame(notebook)

notebook.add(
    stats_tab,
    text="📈 Статистика"
)


average_rating = df[
    "Рейтинг_IMDb"
].mean()

highest_rating = df[
    "Рейтинг_IMDb"
].max()

lowest_rating = df[
    "Рейтинг_IMDb"
].min()

average_duration = df[
    "Длительность_Мин"
].mean()

streaming_count = df[
    "В_Прокате"
].sum()

oscar_count = df[
    "Оскар"
].sum()


stats_text = tk.Text(
    stats_tab,
    font=("Arial", 16),
    padx=30,
    pady=30,
    bg=TEXT_BG,
    fg=FG_COLOR,
    insertbackground=FG_COLOR
)

stats_text.pack(
    fill="both",
    expand=True
)


statistics = f"""
🎬 СТАТИСТИКА ФИЛЬМОВ КИНОТЕАТРА

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎞️ Всего фильмов:
{len(df)}

⭐ Средний рейтинг IMDb:
{average_rating:.2f}

🏆 Наивысший рейтинг IMDb:
{highest_rating:.1f}

📉 Наименьший рейтинг IMDb:
{lowest_rating:.1f}

⏱️ Средняя длительность:
{average_duration:.1f} минут

📺 Доступно в прокате:
{streaming_count}

🏆 Фильмов с Оскаром:
{oscar_count}

📅 Самый старый фильм:
{df.loc[df['Год_Выпуска'].idxmin(), 'Название']}
({df['Год_Выпуска'].min()})

📅 Самый новый фильм:
{df.loc[df['Год_Выпуска'].idxmax(), 'Название']}
({df['Год_Выпуска'].max()})
"""


stats_text.insert(
    tk.END,
    statistics
)

stats_text.config(
    state="disabled"
)


# ============================================================
# ВКЛАДКА 6 - МАТЕМАТИЧЕСКАЯ МОДЕЛЬ
# ============================================================

math_tab = ttk.Frame(notebook)

notebook.add(
    math_tab,
    text="🧮 Математика"
)


math_text = tk.Text(
    math_tab,
    font=("Consolas", 13),
    padx=30,
    pady=30,
    bg=TEXT_BG,
    fg=FG_COLOR,
    insertbackground=FG_COLOR
)

math_text.pack(
    fill="both",
    expand=True
)


math_content = """
КОНТЕНТНАЯ РЕКОМЕНДАЦИЯ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Вектор признаков:

v = [
    Оскар,
    В_Прокате,
    Нормализованный(Год_Выпуска),
    Нормализованный(Рейтинг_IMDb),
    Нормализованный(Длительность)
]


КОСИНУСНОЕ СХОДСТВО

              A · B
Сходство = ─────────
             ||A|| ||B||


Где:

A · B = Σ Ai × Bi

||A|| = √Σ Ai²

||B|| = √Σ Bi²


Значение находится в диапазоне от 0 до 1.

Чем ближе значение к 1,
тем выше сходство между фильмами.


Программа автоматически вычисляет
сходство с помощью:

cosine_similarity()


из библиотеки scikit-learn.


ДОПОЛНИТЕЛЬНО:

Нормализация MinMaxScaler:

       x - min(x)
x' = ───────────────
       max(x) - min(x)


Это приводит все числовые признаки
к диапазону [0, 1].
"""


math_text.insert(
    tk.END,
    math_content
)

math_text.config(
    state="disabled"
)


# ============================================================
# 7. РЕКОМЕНДАЦИИ ПО УМОЛЧАНИЮ
# ============================================================

show_recommendations()


# ============================================================
# 8. ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================

root.mainloop()