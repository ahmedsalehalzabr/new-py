import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 🎬 CINEMA MOVIE RECOMMENDER SYSTEM
# ============================================================

# -----------------------------
# 1. MOVIE DATASET
# -----------------------------

data = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    "Title": [
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Toy Story 4",
        "Parasite",
        "Oppenheimer",
        "Spider-Man: Into the Spider-Verse",
        "Pulp Fiction",
        "Dune: Part Two",
        "Spirited Away"
    ],

    "Has_Oscar": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

    "Is_Streaming": [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],

    "Release_Year": [
        2010, 2014, 2008, 2019, 2019,
        2023, 2018, 1994, 2024, 2001
    ],

    "IMDb_Rating": [
        8.8, 8.7, 9.0, 7.7, 8.5,
        8.9, 8.4, 8.9, 8.6, 8.6
    ],

    "Duration_Min": [
        148, 169, 152, 100, 132,
        180, 117, 154, 166, 125
    ],

    "Genre": [
        "Sci-Fi / Action",
        "Sci-Fi / Drama",
        "Action / Crime",
        "Animation / Family",
        "Drama / Thriller",
        "Biography / Drama",
        "Animation / Action",
        "Crime / Drama",
        "Sci-Fi / Adventure",
        "Animation / Fantasy"
    ],

    "Director": [
        "Christopher Nolan",
        "Christopher Nolan",
        "Christopher Nolan",
        "Josh Cooley",
        "Bong Joon Ho",
        "Christopher Nolan",
        "Bob Persichetti",
        "Quentin Tarantino",
        "Denis Villeneuve",
        "Hayao Miyazaki"
    ]
}

df = pd.DataFrame(data)


# ============================================================
# 2. NORMALIZATION
# ============================================================

numeric_features = [
    "Release_Year",
    "IMDb_Rating",
    "Duration_Min"
]

scaler = MinMaxScaler()

df_scaled = df.copy()

df_scaled[numeric_features] = scaler.fit_transform(
    df[numeric_features]
)


# ============================================================
# 3. FEATURE MATRIX
# ============================================================

features = [
    "Has_Oscar",
    "Is_Streaming",
    "Release_Year",
    "IMDb_Rating",
    "Duration_Min"
]

feature_matrix = df_scaled[features].values


# ============================================================
# 4. COSINE SIMILARITY
# ============================================================

similarity_matrix = cosine_similarity(feature_matrix)


# ============================================================
# 5. RECOMMENDATION FUNCTION
# ============================================================

def get_recommendations(movie_title, top_n=3):

    if movie_title not in df["Title"].values:
        return []

    index = df[df["Title"] == movie_title].index[0]

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
            "Title": df.loc[movie_index, "Title"],
            "Genre": df.loc[movie_index, "Genre"],
            "Director": df.loc[movie_index, "Director"],
            "Rating": df.loc[movie_index, "IMDb_Rating"],
            "Similarity": score * 100
        })

        if len(recommendations) >= top_n:
            break

    return recommendations


# ============================================================
# 6. MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "🎬 Cinema Movie Recommender System"
)

root.geometry("1250x750")

root.minsize(1000, 650)


# ============================================================
# 7. TITLE
# ============================================================

title = tk.Label(
    root,
    text="🎬 CINEMA MOVIE RECOMMENDER SYSTEM",
    font=("Arial", 24, "bold")
)

title.pack(pady=15)


subtitle = tk.Label(
    root,
    text="Recommender Systems & Data Taxonomy",
    font=("Arial", 13)
)

subtitle.pack()


# ============================================================
# 8. NOTEBOOK TABS
# ============================================================

notebook = ttk.Notebook(root)

notebook.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=15
)


# ============================================================
# TAB 1 - DATASET
# ============================================================

dataset_tab = ttk.Frame(notebook)

notebook.add(
    dataset_tab,
    text="📊 Dataset"
)


columns = [
    "ID",
    "Title",
    "Oscar",
    "Streaming",
    "Year",
    "IMDb",
    "Duration",
    "Genre",
    "Director"
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
        width=120
    )


for _, row in df.iterrows():

    tree.insert(
        "",
        "end",
        values=(
            row["ID"],
            row["Title"],
            row["Has_Oscar"],
            row["Is_Streaming"],
            row["Release_Year"],
            row["IMDb_Rating"],
            row["Duration_Min"],
            row["Genre"],
            row["Director"]
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
# TAB 2 - RECOMMENDATIONS
# ============================================================

recommend_tab = ttk.Frame(notebook)

notebook.add(
    recommend_tab,
    text="🎬 Recommendations"
)


label = tk.Label(
    recommend_tab,
    text="Choose a movie:",
    font=("Arial", 15, "bold")
)

label.pack(pady=15)


movie_var = tk.StringVar()

movie_box = ttk.Combobox(
    recommend_tab,
    textvariable=movie_var,
    values=list(df["Title"]),
    width=50,
    state="readonly"
)

movie_box.pack(pady=10)

movie_box.current(0)


result_text = tk.Text(
    recommend_tab,
    font=("Consolas", 12),
    height=20,
    width=100
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
        3
    )

    result_text.delete(
        "1.0",
        tk.END
    )

    result_text.insert(
        tk.END,
        f"🎬 RECOMMENDATIONS FOR: {movie}\n"
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
            f"{i}. {rec['Title']}\n"
        )

        result_text.insert(
            tk.END,
            f"   Genre: {rec['Genre']}\n"
        )

        result_text.insert(
            tk.END,
            f"   Director: {rec['Director']}\n"
        )

        result_text.insert(
            tk.END,
            f"   IMDb Rating: {rec['Rating']}\n"
        )

        result_text.insert(
            tk.END,
            f"   Similarity: {rec['Similarity']:.2f}%\n\n"
        )


button = ttk.Button(
    recommend_tab,
    text="🔎 Generate Recommendations",
    command=show_recommendations
)

button.pack(pady=10)


# ============================================================
# TAB 3 - SIMILARITY MATRIX
# ============================================================

similarity_tab = ttk.Frame(notebook)

notebook.add(
    similarity_tab,
    text="🔢 Similarity Matrix"
)


similarity_tree = ttk.Treeview(
    similarity_tab,
    columns=list(df["Title"]),
    show="headings"
)

for title_name in df["Title"]:

    similarity_tree.heading(
        title_name,
        text=title_name[:12]
    )

    similarity_tree.column(
        title_name,
        width=100
    )


for i, title_name in enumerate(df["Title"]):

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


similarity_tree.pack(
    fill="both",
    expand=True
)


# ============================================================
# TAB 4 - TAXONOMY
# ============================================================

taxonomy_tab = ttk.Frame(notebook)

notebook.add(
    taxonomy_tab,
    text="🌳 Taxonomy"
)


taxonomy_text = tk.Text(
    taxonomy_tab,
    font=("Consolas", 15),
    padx=30,
    pady=30
)

taxonomy_text.pack(
    fill="both",
    expand=True
)


taxonomy = """
                    🎬 CINEMA MOVIES
                           │
             ┌─────────────┴─────────────┐
             │                           │
       🎥 LIVE ACTION               🎨 ANIMATION
             │                           │
      ┌──────┼──────────┐          ┌─────┼──────┐
      │      │          │          │     │      │
   Sci-Fi  Drama     Crime      Family Action Fantasy
      │      │          │          │     │      │
      │      │          │          │     │      │
 Inception Parasite  Pulp      Toy Story Spider  Spirited
 Interstellar Oppenheimer Fiction    4      Verse    Away
 Dune
 Dark Knight
"""


taxonomy_text.insert(
    tk.END,
    taxonomy
)

taxonomy_text.config(
    state="disabled"
)


# ============================================================
# TAB 5 - STATISTICS
# ============================================================

stats_tab = ttk.Frame(notebook)

notebook.add(
    stats_tab,
    text="📈 Statistics"
)


average_rating = df[
    "IMDb_Rating"
].mean()

highest_rating = df[
    "IMDb_Rating"
].max()

lowest_rating = df[
    "IMDb_Rating"
].min()

average_duration = df[
    "Duration_Min"
].mean()

streaming_count = df[
    "Is_Streaming"
].sum()

oscar_count = df[
    "Has_Oscar"
].sum()


stats_text = tk.Text(
    stats_tab,
    font=("Arial", 16),
    padx=30,
    pady=30
)

stats_text.pack(
    fill="both",
    expand=True
)


statistics = f"""
🎬 CINEMA MOVIE STATISTICS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎞️ Total Movies:
{len(df)}

⭐ Average IMDb Rating:
{average_rating:.2f}

🏆 Highest IMDb Rating:
{highest_rating:.1f}

📉 Lowest IMDb Rating:
{lowest_rating:.1f}

⏱️ Average Duration:
{average_duration:.1f} minutes

📺 Available for Streaming:
{streaming_count}

🏆 Movies with Oscar:
{oscar_count}

📅 Oldest Movie:
{df.loc[df['Release_Year'].idxmin(), 'Title']}
({df['Release_Year'].min()})

📅 Newest Movie:
{df.loc[df['Release_Year'].idxmax(), 'Title']}
({df['Release_Year'].max()})
"""


stats_text.insert(
    tk.END,
    statistics
)

stats_text.config(
    state="disabled"
)


# ============================================================
# TAB 6 - MATHEMATICAL MODEL
# ============================================================

math_tab = ttk.Frame(notebook)

notebook.add(
    math_tab,
    text="🧮 Mathematics"
)


math_text = tk.Text(
    math_tab,
    font=("Consolas", 14),
    padx=30,
    pady=30
)

math_text.pack(
    fill="both",
    expand=True
)


math_content = """
CONTENT-BASED RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Feature Vector:

v = [
    Has_Oscar,
    Is_Streaming,
    Normalized(Release_Year),
    Normalized(IMDb_Rating),
    Normalized(Duration)
]


COSINE SIMILARITY

              A · B
Similarity = ─────────
             ||A|| ||B||


Where:

A · B = Σ Ai × Bi

||A|| = √Σ Ai²

||B|| = √Σ Bi²


The value is between 0 and 1.

A value closer to 1 means
greater similarity between movies.


The program calculates the similarity
automatically using:

cosine_similarity()


from scikit-learn.
"""


math_text.insert(
    tk.END,
    math_content
)

math_text.config(
    state="disabled"
)


# ============================================================
# 7. DEFAULT RECOMMENDATIONS
# ============================================================

show_recommendations()


# ============================================================
# 8. RUN APPLICATION
# ============================================================

root.mainloop()