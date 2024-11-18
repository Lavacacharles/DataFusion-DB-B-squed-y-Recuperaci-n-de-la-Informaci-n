#!/usr/bin/env python3

import psycopg2
import pandas as pd
import numpy as np
from preprocesamiento import preprocesamiento

conn = psycopg2.connect(
    database="bd2_proyecto",
    host="localhost",
    user="postgres",
    password="panza",
    port="5432",
)


# Create table
def create_table():
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXITS songs (
                track_id varchar(22),
                lyrics varchar(27698),
                track_name varchar(123),
                track_artist varchar(51),
                track_album_name varchar(151),
                playlist_name varchar(120),
                playlist_genre varchar(5),
                playlist_subgenre varchar(25),
                language varchar(2),
                info_vector tsvector
        );
        """
        )
        print("Execution done")
        cursor.close()
        print("Cursor closed")
        conn.commit()
        print("Changes commited")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def insert_all():
    try:
        cursor = conn.cursor()

        cursor.execute("SELEECT COUNT(*) FROM songs")

        if cursor.fetchone()["count"] == 0:
            columns = [
                "track_id",
                "lyrics",
                "track_name",
                "track_artist",
                "track_album_name",
                "playlist_name",
                "playlist_genre",
                "playlist_subgenre",
                "language",
            ]
            df = pd.read_csv("songs.csv")
            df["language"] = df["language"].replace({np.nan: None})
            command = """
            INSERT INTO songs(track_id, lyrics, track_name,
                          track_artist, track_album_name, playlist_name,
                          playlist_genre, playlist_subgenre, language)
            VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            for _, row in df.iterrows():
                cursor.execute(command, tuple(row[col] for col in columns))
            print("Execution done")
            print("Cursor closed")
            conn.commit()
            print("Changes commited")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        conn.rollback()
    finally:
        cursor.close()


def set_index():
    try:
        cursor = conn.cursor()

        command = """
        CREATE INDEX IF NOT EXISTS idx_info_vector ON
        songs USING GIN(info_vector)
        """
        cursor.execute(command)
        print("Execution done")
        print("Cursor closed")
        conn.commit()
        print("Changes commited")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        conn.rollback()
        conn.rollback()
    finally:
        cursor.close()


def update_index():
    try:
        cursor = conn.cursor()

        command = """
        UPDATE songs SET info_vector =
            setweight(to_tsvector('english', COALESCE(track_name, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(track_album_name, '')), 'B') ||
            setweight(to_tsvector('english', COALESCE(track_artist, '')), 'C') ||
            setweight(to_tsvector('english', COALESCE(lyrics, '')), 'D');
        """
        cursor.execute(command)
        print("Execution done")
        print("Cursor closed")
        conn.commit()
        print("Changes commited")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        conn.rollback()
        conn.rollback()
    finally:
        cursor.close()


def clean():
    try:
        cursor = conn.cursor()

        command = """
        DELETE FROM songs;
        """
        cursor.execute(command)
        print("Execution done")
        print("Cursor closed")
        conn.commit()
        print("Changes commited")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        conn.rollback()
        conn.rollback()
    finally:
        cursor.close()


def extract_time(time):
    return float(time[-1][0].split(":")[1][1:].split(" ")[0])


def search(query, columns, k=10, time=False):
    try:
        # The query is procesed to put the or operators
        query_procesed = preprocesamiento(query)

        # Parse columns list so that they are a string of the selected
        columns_parsed = ", ".join(columns) + ", "

        # Parse columns list so that they are a string of the selected

        cursor = conn.cursor()

        command = f"""
        SELECT {columns_parsed}
               ctid::text as row_position,
               ts_rank_cd(info_vector, {query_procesed}) as similitud
        FROM songs, to_tsquery({query_procesed}) query
        WHERE info_vector @@ query
        ORDER BY similitud DESC
        LIMIT {k};
        """
        if time is False:
            cursor.execute(command)
            results = cursor.fetchall()
        else:
            command = "EXPLAIN ANALYZE " + command
            cursor.execute(command)
            results = cursor.fetchall()
            results = extract_time(results)
        print("Execution done")
        # returns a list with tuples, where every tuple is a row
        return results
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        conn.rollback()
        conn.rollback()
    finally:
        cursor.close()


# Use if first time creating
# create_table()
# clean()
# insert()
# set_index()
# update_index()

columns = ["track_id", "track_name", "track_artist", "track_album_name"]
results = search("Queen is dead, She is not a alive, girl", columns, 10, False)
# for i in results:
#     print(i)
