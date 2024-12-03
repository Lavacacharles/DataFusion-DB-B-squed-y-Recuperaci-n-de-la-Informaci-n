#!/usr/bin/env python3

import psycopg2
import pandas as pd
import numpy as np
from preprocesamiento import preprocesamiento


# Put your credentials
conn = psycopg2.connect(
    database="bd2_proyecto",
    host="localhost",
    user="postgres",
    password="postgres",
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

        cursor.close()
        conn.commit()
        print("Table created")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def insert_all(csv_path, columns):
    try:
        cursor = conn.cursor()

        columns_parsed = ", ".join(columns)

        cursor.execute("SELECT COUNT(*) FROM songs")
        print("Cursor created")

        count = cursor.fetchone()
        count = count[0]
        if count == 0:
            df = pd.read_csv(csv_path)
            df["language"] = df["language"].replace({np.nan: None})
            command = f"""
            INSERT INTO songs({columns_parsed})
            VALUES(%s{ ", %s" * (len(columns) - 1)});
            """
            for _, row in df.iterrows():
                values = tuple(row[col] for col in columns)
                cursor.execute(command, values)

            conn.commit()
            print("Insertion commited")
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: ", error)
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

        conn.commit()
        print("Index created")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        conn.rollback()
        conn.rollback()
    finally:
        cursor.close()


def update_index(language):
    try:
        cursor = conn.cursor()

        command = f"""
        UPDATE songs SET info_vector =
            setweight(to_tsvector('{language}', COALESCE(track_name, '')), 'A') ||
            setweight(to_tsvector('{language}', COALESCE(track_album_name, '')), 'B') ||
            setweight(to_tsvector('{language}', COALESCE(track_artist, '')), 'C') ||
            setweight(to_tsvector('{language}', COALESCE(lyrics, '')), 'D');
        """
        cursor.execute(command)

        conn.commit()
        print("Infor vector updated")
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

        conn.commit()
        print("Table empty")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        conn.rollback()
        conn.rollback()
    finally:
        cursor.close()


def extract_time(time):
    """
    Given the fechall of a explain analyze, it returns the time
    """
    return float(time[-1][0].split(":")[1][1:].split(" ")[0])


def search(query, columns, k=10, time=False):
    """
    if time is true, the executed time of the command is returned
    """
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
               ts_rank_cd(info_vector, {query_procesed}) as score
        FROM songs, to_tsquery({query_procesed}) query
        WHERE info_vector @@ query
        ORDER BY score DESC
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
# set_index()

# The below code is for inserting the data
clean()
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





# The csv file needs to have only the above columns
# To generate this csv, use the clean.py file
# Currently the songs_20.csv and songs.csv datasets are cleaned.
# Found in the datasets directory
csv_path = "datasets/spotify_1000.csv"
insert_all(csv_path, columns)
update_index("english")


def obtain_times(queries, columns, k):
    times = []
    for i in queries:
        current_time = search(i, columns, k, True)
        print(current_time)
        times.append(current_time)
    return times


columns = ["track_id", "track_name", "track_artist", "track_album_name"]
# Las queries deben estar asi porque con """ hay más espacio, arruinando el
# preprocesamiento
queries = [
    "Don't sweat all the little things Just keep your eye on the bigger things Cause if you look a little closer You're gonna get a bigger picture",
    "I'mma make your CXRPSE dance Ugh, hop in that Jag, do the dash I shoot a nigga then laugh Bitch, don't talk to me if you ain't on that",
]


k = 20
times = obtain_times(queries, columns, k)
for i in range(len(times)):
    print("Querry i: ", queries[i])
    print("Elapsed time: ", times[i], "ms")