#!/usr/bin/env python3

import pandas as pd

df = pd.read_csv("spotify_songs.csv")
df_selected = df.loc[
    :,
    [
        "track_id",
        "lyrics",
        "track_name",
        "track_artist",
        "track_album_name",
        "playlist_name",
        "playlist_genre",
        "playlist_subgenre",
        "language",
    ],
]
print(df_selected)
df_selected.to_csv("songs.csv", index=False)
maximos = []
for column_name, value in df_selected.items():
    # print(column_name)
    max_length = 0
    for row in value:
        # print(row)
        if not pd.isna(row):
            max_length = max(max_length, len(row))
    maximos.append((column_name, max_length))

for i in maximos:
    print(f"{i[0]} varchar({i[1]}),")
