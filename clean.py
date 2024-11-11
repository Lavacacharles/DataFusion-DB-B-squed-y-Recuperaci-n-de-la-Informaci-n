#!/usr/bin/env python3

import pandas as pd

df = pd.read_csv("spotify_songs.csv")
df_selected = df.loc[df["language"] == "en", ["lyrics"]]
print(df_selected)
df_selected.to_csv("lyrics.csv", index_label="id")
