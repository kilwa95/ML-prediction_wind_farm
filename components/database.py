import mysql.connector
import streamlit as st

# Fonction pour initialiser la connexion
def init_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"],
        port=st.secrets["mysql"]["port"]
    )

# Fonction pour exécuter des requêtes
def run_query(query, params=None):
    conn = init_connection()
    with conn.cursor() as cur:
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        if query.lower().startswith('select'):
            return cur.fetchall()
        else:
            conn.commit()
            return cur.rowcount
    conn.close() 