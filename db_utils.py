# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import streamlit as st
import traceback
import os
from contextlib import contextmanager
import numpy as np

# --- Costanti ---
DB_FILENAME = "gare_appalto.db"
EXPECTED_COLUMNS = [
    'id', 'identificativo_gara', 'descrizione', 'data_gara', 'importo_base',
    'categoria_lavori', 'stazione_appaltante', 'mio_ribasso_percentuale',
    'importo_offerto', 'soglia_anomalia_calcolata', 'ribasso_aggiudicatario_percentuale',
    'importo_aggiudicazione', 'numero_concorrenti', 'posizione_in_graduatoria',
    'esito', 'note', 'data_inserimento'
]

# --- Gestione Connessione ---
@st.cache_resource(ttl=3600)
def init_connection():
    """Inizializza e ritorna la connessione al DB, gestendo errori."""
    try:
        if not os.path.exists(DB_FILENAME): print(f"DB '{DB_FILENAME}' non trovato. Creo.")
        conn = sqlite3.connect(DB_FILENAME, check_same_thread=False, timeout=15.0)
        conn.row_factory = sqlite3.Row # Permette accesso per nome colonna
        print("Connessione DB inizializzata."); return conn
    except sqlite3.Error as e: print(f"Errore SQLite conn: {e}"); st.error(f"Errore critico DB: {e}"); traceback.print_exc(); return None

@contextmanager
def get_db_cursor():
    """Fornisce un cursore DB gestendo commit/rollback."""
    conn = init_connection();
    if conn is None: yield None; return # Esce subito se non c'è connessione
    cursor = None
    try:
        cursor = conn.cursor(); yield cursor; conn.commit()
    except sqlite3.Error as e:
        print(f"Errore DB durante operazione: {e}. Eseguo Rollback.");
        if conn:
            try: conn.rollback(); print("Rollback eseguito.")
            except Exception as rb_err: print(f"Errore durante rollback: {rb_err}")
        # Rilancia l'eccezione per segnalare il fallimento all'esterno
        raise e
    except Exception as e:
        print(f"Errore imprevisto context DB: {e}. Eseguo Rollback.")
        if conn:
            try: conn.rollback(); print("Rollback eseguito.")
            except Exception as rb_err: print(f"Errore durante rollback: {rb_err}")
        raise e
    finally:
        # Non chiudiamo conn qui, è gestita da cache_resource
        # Il cursore viene chiuso implicitamente uscendo dal 'with' nel chiamante
        pass


# --- Funzioni CRUD ---
def create_table():
    """Crea tabella e indici se non esistono."""
    try:
        with get_db_cursor() as cursor:
            if cursor is None: print("Impossibile creare tabella: no cursor."); return
            # Definisci tipi colonne più specifici e vincoli
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gare (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identificativo_gara TEXT UNIQUE NOT NULL,
                    descrizione TEXT,
                    data_gara DATE,
                    importo_base REAL,
                    categoria_lavori TEXT,
                    stazione_appaltante TEXT,
                    mio_ribasso_percentuale REAL,
                    importo_offerto REAL,
                    soglia_anomalia_calcolata REAL,
                    ribasso_aggiudicatario_percentuale REAL,
                    importo_aggiudicazione REAL,
                    numero_concorrenti INTEGER,
                    posizione_in_graduatoria INTEGER, -- NULL se non in graduatoria
                    esito TEXT,
                    note TEXT,
                    data_inserimento TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
                )""")
            # Aggiungi indici per colonne usate frequentemente nei filtri/join
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_gara ON gare (data_gara);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_categoria ON gare (categoria_lavori);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_esito ON gare (esito);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_importo_base ON gare (importo_base);") # Indice su importo per filtri range
            print("Tabella 'gare' e indici verificati/creati con successo.")
    except Exception as e: print(f"Errore durante create_table: {e}"); traceback.print_exc()

def add_gara(data: dict) -> bool:
    """Aggiunge una nuova gara al database."""
    if 'identificativo_gara' not in data or not str(data['identificativo_gara']).strip():
        print("Errore add_gara: CIG mancante o vuoto."); return False

    # Pulisci dati: prendi solo colonne attese, converti NaN/None in NULL per SQL
    clean_data = {}
    for col in EXPECTED_COLUMNS:
        if col in data and col not in ['id', 'data_inserimento']: # Escludi ID e data inserimento (auto-gestiti)
            value = data[col]
            # Converti pd.NA, np.nan, None, '' in None (che diventa NULL in SQL)
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                clean_data[col] = None
            # Gestisci specificamente 0 per posizione_in_graduatoria come NULL
            elif col == 'posizione_in_graduatoria' and value == 0:
                 clean_data[col] = None
            else:
                clean_data[col] = value

    if not clean_data:
        print("Errore add_gara: Nessun dato valido da inserire."); return False

    columns = ', '.join(clean_data.keys())
    placeholders = ', '.join([f":{key}" for key in clean_data.keys()])
    sql = f'INSERT INTO gare ({columns}) VALUES ({placeholders})'

    try:
        with get_db_cursor() as cursor:
            if cursor is None: return False
            cursor.execute(sql, clean_data)
            print(f"Gara '{clean_data.get('identificativo_gara', 'N/A')}' aggiunta con successo (ID: {cursor.lastrowid})."); return True
    except sqlite3.IntegrityError:
        # Errore specifico per violazione vincolo UNIQUE (CIG duplicato)
        st.error(f"Errore: Esiste già una gara con CIG '{clean_data.get('identificativo_gara', 'N/A')}'.")
        print(f"Errore Integrità (CIG Duplicato?): Gara '{clean_data.get('identificativo_gara', 'N/A')}'")
        return False
    except sqlite3.Error as e:
        # Altri errori SQL
        st.error(f"Errore database durante l'inserimento: {e}")
        print(f"Errore SQL add_gara: {e}"); traceback.print_exc(); return False
    except Exception as e:
        # Errori generici
        st.error(f"Errore imprevisto durante l'inserimento: {e}")
        print(f"Errore generico add_gara: {e}"); traceback.print_exc(); return False

@st.cache_data(ttl=300) # Cache per 5 minuti
def get_all_gare(_refresh_trigger=None) -> pd.DataFrame:
    """Recupera tutte le gare dal database come DataFrame pandas."""
    print(f"get_all_gare chiamato con trigger: {_refresh_trigger}") # Utile per debug cache
    conn = init_connection();
    if not conn: return pd.DataFrame(columns=EXPECTED_COLUMNS) # Ritorna DF vuoto se connessione fallisce

    try:
        # Ordina per data più recente prima, poi per ID decrescente come fallback
        query = "SELECT * FROM gare ORDER BY data_gara DESC, id DESC"
        df = pd.read_sql_query(query, conn)

        # Conversioni di tipo post-lettura per coerenza
        if 'data_gara' in df.columns: df['data_gara'] = pd.to_datetime(df['data_gara'], errors='coerce')
        if 'data_inserimento' in df.columns: df['data_inserimento'] = pd.to_datetime(df['data_inserimento'], errors='coerce')

        # Colonne numeriche (float)
        float_cols = ['importo_base', 'mio_ribasso_percentuale', 'importo_offerto', 'soglia_anomalia_calcolata', 'ribasso_aggiudicatario_percentuale', 'importo_aggiudicazione']
        for col in float_cols:
             if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        # Colonne numeriche (integer, gestendo NaN)
        int_cols = ['numero_concorrenti', 'posizione_in_graduatoria']
        for col in int_cols:
              if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64') # Int64 supporta NaN

        print(f"Recuperate {len(df)} gare dal database."); return df
    except Exception as e:
        # Gestisce errori durante la lettura o conversione
        st.error(f"Errore durante il recupero delle gare: {e}")
        print(f"Errore in get_all_gare: {e}"); traceback.print_exc()
        return pd.DataFrame(columns=EXPECTED_COLUMNS) # Ritorna DF vuoto in caso di errore

@st.cache_data(ttl=60) # Cache più breve per dati specifici
def get_gara_by_id(gara_id: int, _cache_key_modifier=None) -> dict | None:
    """Recupera una singola gara per ID."""
    print(f"get_gara_by_id chiamato per ID: {gara_id}, modifier: {_cache_key_modifier}")
    # Validazione input ID
    if not isinstance(gara_id, (int, np.integer)) or gara_id <= 0:
        print(f"ID gara non valido fornito: {gara_id}"); return None

    try:
        with get_db_cursor() as cursor:
            if cursor is None: return None
            # Usa parameterized query per sicurezza
            cursor.execute("SELECT * FROM gare WHERE id = ?", (int(gara_id),))
            gara_data = cursor.fetchone() # fetchone ritorna una riga (Row object) o None
            if gara_data:
                print(f"Recuperata gara con ID {gara_id}.");
                return dict(gara_data) # Converti Row object in dict
            else:
                print(f"Nessuna gara trovata con ID {gara_id}.");
                return None
    except sqlite3.Error as e:
        st.error(f"Errore database recuperando gara ID {gara_id}: {e}")
        print(f"Errore SQL in get_gara_by_id: {e}"); return None
    except Exception as e:
        st.error(f"Errore imprevisto recuperando gara ID {gara_id}: {e}")
        print(f"Errore generico in get_gara_by_id: {e}"); return None

def update_gara(gara_id: int, data: dict) -> bool:
    """Aggiorna una gara esistente nel database."""
    # Validazione ID
    if not isinstance(gara_id, (int, np.integer)) or gara_id <= 0:
         print(f"Errore update_gara: ID gara non valido ({gara_id})."); return False

    # Colonne che possono essere aggiornate (escludi ID, CIG, data inserimento)
    allowed_to_update = [col for col in EXPECTED_COLUMNS if col not in ['id', 'identificativo_gara', 'data_inserimento']]

    # Prepara i dati per l'aggiornamento: includi solo colonne permesse
    # e converti valori "vuoti" in None per sovrascrivere con NULL nel DB
    update_data = {}
    for col in allowed_to_update:
        if col in data: # Controlla se la colonna è presente nei dati forniti
            value = data[col]
            # Se il valore è NaN, None o stringa vuota, imposta a None (SQL NULL)
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                 update_data[col] = None
            # Gestisci 0 per posizione come NULL
            elif col == 'posizione_in_graduatoria' and value == 0:
                 update_data[col] = None
            else:
                 update_data[col] = value

    if not update_data:
        print(f"Nessun dato valido fornito per aggiornare Gara ID {gara_id}."); return False # Nessun campo valido da aggiornare

    # Costruisci la clausola SET dinamicamente
    set_clause = ', '.join([f"{key} = :{key}" for key in update_data.keys()])
    sql = f"UPDATE gare SET {set_clause} WHERE id = :id"

    # Aggiungi l'ID al dizionario dei dati per il binding
    update_data['id'] = int(gara_id)

    try:
        with get_db_cursor() as cursor:
            if cursor is None: return False
            cursor.execute(sql, update_data)
            if cursor.rowcount > 0:
                print(f"Gara ID {gara_id} aggiornata con successo ({cursor.rowcount} riga/e modificata/e).")
                # Invalida cache specifiche dopo modifica
                get_gara_by_id.clear()
                get_all_gare.clear() # Modifica potrebbe impattare la lista completa
                return True
            else:
                # Nessuna riga modificata: o l'ID non esiste o i dati erano identici
                print(f"Nessuna riga aggiornata per Gara ID {gara_id}. La gara esiste e i dati sono invariati, oppure l'ID non esiste.")
                # Potremmo voler verificare se l'ID esiste per distinguere i casi
                # gara_esiste = get_gara_by_id(gara_id) is not None
                # print(f"La gara ID {gara_id} {'esiste' if gara_esiste else 'non esiste'}.")
                return False # Consideriamo False se non ci sono state modifiche effettive
    except sqlite3.Error as e:
        st.error(f"Errore database durante l'aggiornamento Gara ID {gara_id}: {e}")
        print(f"Errore SQL update_gara: {e}"); traceback.print_exc(); return False
    except Exception as e:
        st.error(f"Errore imprevisto durante l'aggiornamento Gara ID {gara_id}: {e}")
        print(f"Errore generico update_gara: {e}"); traceback.print_exc(); return False


def delete_gara_by_id(gara_id: int) -> bool:
    """Elimina una gara specifica dal database per ID."""
     # Validazione ID
    if not isinstance(gara_id, (int, np.integer)) or gara_id <= 0:
        print(f"Errore delete_gara_by_id: ID gara non valido ({gara_id})."); return False

    sql = 'DELETE FROM gare WHERE id = ?'
    try:
        with get_db_cursor() as cursor:
            if cursor is None: return False
            cursor.execute(sql, (int(gara_id),)) # Usa tupla per parametri posizionali
            if cursor.rowcount > 0:
                print(f"Gara ID {gara_id} eliminata con successo ({cursor.rowcount} riga/e).")
                # Invalida cache dopo eliminazione
                get_gara_by_id.clear() # Rimuovi eventuale cache per questo ID
                get_all_gare.clear() # La lista completa è cambiata
                return True
            else:
                print(f"Nessuna gara trovata con ID {gara_id} per l'eliminazione.");
                return False # ID non trovato
    except sqlite3.Error as e:
        st.error(f"Errore database durante l'eliminazione Gara ID {gara_id}: {e}")
        print(f"Errore SQL delete_gara_by_id: {e}"); traceback.print_exc(); return False
    except Exception as e:
        st.error(f"Errore imprevisto durante l'eliminazione Gara ID {gara_id}: {e}")
        print(f"Errore generico delete_gara_by_id: {e}"); traceback.print_exc(); return False

def clear_all_cache():
    """Pulisce tutte le cache dati DB conosciute definite con @st.cache_data."""
    print("Tentativo pulizia cache Streamlit @st.cache_data...")
    try:
        get_all_gare.clear()
        print("- Cache get_all_gare pulita.")
    except Exception as e:
        print(f"- Errore pulizia cache get_all_gare: {e}")
    try:
        get_gara_by_id.clear()
        print("- Cache get_gara_by_id pulita.")
    except Exception as e:
        print(f"- Errore pulizia cache get_gara_by_id: {e}")
    # Aggiungi qui altre funzioni cachate se necessario
    print("Pulizia cache Streamlit completata.")

# --- Inizializzazione ---
# Assicura che la tabella esista all'avvio dell'applicazione
create_table()