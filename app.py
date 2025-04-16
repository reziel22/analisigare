# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import datetime
import traceback
import os
import numpy as np

# --- Configurazione Pagina ---
st.set_page_config(page_title="Analisi Gare Appalto v1.7", layout="wide", initial_sidebar_state="expanded")

# Importa moduli custom
import db_utils
import ml_utils

# --- Costanti di Formattazione ---
# Usate per input e logica interna (standard float)
PERCENTAGE_FORMAT = "%.4f"
CURRENCY_INTERNAL_FORMAT = "%.2f"
# Usate per display in tabelle (con simboli)
PERCENTAGE_DISPLAY_FORMAT = "%.4f%%" # Aggiunge '%'
CURRENCY_COLUMN_FORMAT = "€ %.2f" # Aggiunge '€ ' (spazio importante)
# Usate per display in st.metric (f-string style, con simboli)
PERCENTAGE_METRIC_FORMAT = "{:.4f}%"
# Usate per display date/datetime
DATE_FORMAT_STR = "%Y-%m-%d" # Formato per DB/pandas
DATETIME_FORMAT_STR = "%d/%m/%Y %H:%M" # Formato display
DATE_DISPLAY_FORMAT = "DD/MM/YYYY" # Formato display date-only

# --- Gestione Stato Sessione ---
# (Stato Sessione INVARIATO)
if 'refresh_data' not in st.session_state: st.session_state.refresh_data = 0
form_widget_keys = ["widget_identificativo_gara", "widget_descrizione", "widget_data_gara", "widget_importo_base", "widget_categoria_lavori", "widget_stazione_appaltante", "widget_mio_ribasso_percentuale", "widget_soglia_anomalia_calcolata", "widget_ribasso_aggiudicatario_percentuale", "widget_numero_concorrenti", "widget_posizione_in_graduatoria", "widget_esito", "widget_note"]
default_form_values = {"widget_identificativo_gara": "", "widget_descrizione": "", "widget_data_gara": None, "widget_importo_base": None, "widget_categoria_lavori": "", "widget_stazione_appaltante": "", "widget_mio_ribasso_percentuale": None, "widget_soglia_anomalia_calcolata": None, "widget_ribasso_aggiudicatario_percentuale": None, "widget_numero_concorrenti": None, "widget_posizione_in_graduatoria": None, "widget_esito": "", "widget_note": ""}
for key in form_widget_keys:
    if key not in st.session_state: st.session_state[key] = default_form_values[key]
if 'form_submit_success' not in st.session_state: st.session_state.form_submit_success = False
if 'editing_gara_id' not in st.session_state: st.session_state.editing_gara_id = None
edit_form_keys = [f"edit_{k.replace('widget_', '')}" for k in form_widget_keys]
for key in edit_form_keys:
    if key not in st.session_state: st.session_state[key] = None
if 'edit_form_reset_needed' not in st.session_state: st.session_state.edit_form_reset_needed = False

def trigger_data_refresh():
    st.session_state.refresh_data += 1
    db_utils.clear_all_cache()

# --- Funzioni App ---
#@st.cache_data # Caching può essere utile ma attenti con oggetti file
def load_data_from_file(uploaded_file):
    """Carica dati da file CSV o Excel, pulisce e mappa le colonne."""
    try:
        file_name = uploaded_file.name
        df = None
        read_success = False
        st.write(f"Lettura file: {file_name}")

        # --- Lettura File ---
        if file_name.endswith('.csv'):
            # Prova combinazioni comuni di encoding, separatori, decimali
            common_encodings=['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            common_separators=[None, ';', ','] # None = auto-detect
            common_decimals=[',', '.']
            for enc in common_encodings:
                for sep in common_separators:
                    for dec in common_decimals:
                        try:
                            uploaded_file.seek(0) # Torna all'inizio del file
                            df_temp = pd.read_csv(uploaded_file, sep=sep, engine='python', encoding=enc, decimal=dec)
                            # Verifica se la lettura ha prodotto colonne significative
                            if df_temp is not None and df_temp.shape[1] > 1: # Più di una colonna è un buon segno
                                df = df_temp
                                st.write(f"Letto CSV con successo: sep='{sep if sep else 'auto'}', encoding='{enc}', decimal='{dec}'")
                                read_success = True
                                break # Esce dal loop decimali
                        except Exception as e_read:
                            #st.write(f"Tentativo CSV fallito: sep={sep}, enc={enc}, dec={dec}, Err: {e_read}") # Debug
                            continue # Prova prossima combinazione
                    if read_success: break # Esce dal loop separatori
                if read_success: break # Esce dal loop encoding
            if not read_success or df is None or df.empty:
                st.error("Lettura CSV fallita con tutte le combinazioni comuni. Verifica formato, encoding, separatore e decimale del file.")
                return None
        elif file_name.endswith(('.xls', '.xlsx')):
            try:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                st.success("File Excel caricato con successo.")
            except Exception as e_excel:
                st.error(f"Errore durante la lettura del file Excel: {e_excel}")
                st.error(traceback.format_exc())
                return None
        else:
            st.error("Formato file non supportato. Caricare un file .csv, .xls o .xlsx.")
            return None

        # --- Mappatura e Pulizia Colonne ---
        st.write("Colonne originali:", df.columns.tolist())
        column_mapping = {
            # Mappature CIG (fondamentale)
            'cig': 'identificativo_gara', 'id gara': 'identificativo_gara',
            # Mappature Descrizione
            'oggetto gara': 'descrizione', 'oggetto': 'descrizione', 'descrizione gara': 'descrizione',
            # Mappature Data
            'data scadenza': 'data_gara', 'data': 'data_gara', 'data pubblicazione': 'data_gara',
            # Mappature Importo Base
            'importo a base d\'asta': 'importo_base', 'importo': 'importo_base', 'base asta': 'importo_base', 'importo base': 'importo_base',
            # Mappature Categoria
            'categoria': 'categoria_lavori', 'cat.': 'categoria_lavori', 'categoria prevalente': 'categoria_lavori',
            # Mappature Stazione Appaltante
            'stazione appaltante': 'stazione_appaltante', 'sa': 'stazione_appaltante', 'ente': 'stazione_appaltante',
            # Mappature Ribasso Offerto (%)
            'nostro ribasso (%)': 'mio_ribasso_percentuale', 'ribasso offerto %': 'mio_ribasso_percentuale', 'tuo ribasso %': 'mio_ribasso_percentuale', 'mio ribasso %': 'mio_ribasso_percentuale',
            # Mappature Importo Offerto (€) - Calcolato se non presente
            'nostra offerta': 'importo_offerto', 'importo offerto': 'importo_offerto',
            # Mappature Soglia Anomalia (%)
            'soglia anomalia (%)': 'soglia_anomalia_calcolata', 'soglia %': 'soglia_anomalia_calcolata', 'soglia di anomalia %': 'soglia_anomalia_calcolata',
            # Mappature Ribasso Aggiudicatario (%)
            'ribasso aggiudicatario (%)': 'ribasso_aggiudicatario_percentuale', 'ribasso agg. %': 'ribasso_aggiudicatario_percentuale', 'ribasso aggiudicazione %': 'ribasso_aggiudicatario_percentuale',
            # Mappature Importo Aggiudicazione (€) - Calcolato se non presente
            'importo aggiudicazione': 'importo_aggiudicazione', 'importo agg.': 'importo_aggiudicazione', 'importo aggiudicato': 'importo_aggiudicazione',
            # Mappature Numero Concorrenti
            'num concorrenti': 'numero_concorrenti', 'num. offerte': 'numero_concorrenti', 'numero offerte': 'numero_concorrenti',
            # Mappature Posizione Graduatoria
            'posizione graduatoria': 'posizione_in_graduatoria', 'posizione': 'posizione_in_graduatoria', 'ns posizione': 'posizione_in_graduatoria',
            # Mappature Esito
            'esito gara': 'esito', 'esito': 'esito', 'stato': 'esito',
            # Mappature Note
            'annotazioni': 'note', 'note': 'note'
        }
        df.columns = df.columns.str.lower().str.strip() # Pulisci nomi colonne originali
        df.rename(columns=column_mapping, inplace=True)
        st.write("Colonne dopo mappatura preliminare:", df.columns.tolist())

        # --- Conversione Tipi Numerici ---
        # Lista colonne target che devono essere numeriche nel DB
        numeric_cols_db = ['importo_base', 'mio_ribasso_percentuale', 'importo_offerto',
                           'soglia_anomalia_calcolata', 'ribasso_aggiudicatario_percentuale',
                           'importo_aggiudicazione', 'numero_concorrenti', 'posizione_in_graduatoria']
        for col in numeric_cols_db:
            if col in df.columns:
                # Salva originale per debug
                df[col+'_original'] = df[col]
                # Converti a stringa, pulisci simboli (€, %), spazi
                # Gestisci separatore migliaia (.) e decimale (,) -> converti a standard (.)
                cleaned_series = df[col].astype(str).str.replace('€', '', regex=False)\
                                    .str.replace('%', '', regex=False)\
                                    .str.replace(r'\s+', '', regex=True)\
                                    .str.strip()
                # Prima rimuovi punti (migliaia), poi sostituisci virgola (decimale) con punto
                standardized_series = cleaned_series.str.replace(r'\.(?=.*\.)', '', regex=True) # Rimuove punti tranne l'ultimo (se presente)
                # Caso speciale: se c'è solo un punto e nessuna virgola, è prob. decimale
                mask_single_dot = standardized_series.str.count(r'\.') == 1 & ~standardized_series.str.contains(',')
                # Caso generale: sostituisci virgola con punto
                standardized_series = standardized_series.str.replace(',', '.', regex=False)

                df[col] = pd.to_numeric(standardized_series, errors='coerce')

                # Warning se la conversione fallisce per valori non vuoti
                failed_numeric = df[df[col].isna() & df[col+'_original'].notna() & (df[col+'_original'].astype(str).str.strip() != '')]
                if not failed_numeric.empty:
                    st.warning(f"Conversione numerica fallita per alcuni valori in '{col}'. Esempi non convertiti: {failed_numeric[col+'_original'].unique()[:5]}")
            #else:
            #    st.write(f"Colonna numerica attesa '{col}' non trovata nel file.")

        # --- Conversione Data ---
        if 'data_gara' in df.columns:
            df['data_gara_original'] = df['data_gara']
            # Prova il formato DD/MM/YYYY prima (comune in Italia)
            df['data_gara'] = pd.to_datetime(df['data_gara'], errors='coerce', dayfirst=True)
            # Se fallisce, prova il formato standard YYYY-MM-DD o altri riconosciuti da pandas
            mask_failed = df['data_gara'].isna() & df['data_gara_original'].notna()
            if mask_failed.any():
                df.loc[mask_failed, 'data_gara'] = pd.to_datetime(df.loc[mask_failed, 'data_gara_original'], errors='coerce')
            # Warning se ancora NaN
            failed_dates = df[df['data_gara'].isna() & df['data_gara_original'].notna() & (df['data_gara_original'] != '')]
            if not failed_dates.empty:
                st.warning(f"Conversione data fallita per alcuni valori in 'data_gara'. Esempi non convertiti: {failed_dates['data_gara_original'].unique()[:5]}")
        #else:
        #    st.write("Colonna 'data_gara' non trovata nel file.")


        # --- Calcoli Derivati (se mancano colonne) ---
        # Calcola importo_offerto se mancano ma ci sono base e ribasso
        if 'importo_offerto' not in df.columns and ('importo_base' in df.columns and 'mio_ribasso_percentuale' in df.columns):
            mask = df['importo_base'].notna() & df['mio_ribasso_percentuale'].notna()
            df.loc[mask, 'importo_offerto'] = df.loc[mask, 'importo_base'] * (1 - df.loc[mask, 'mio_ribasso_percentuale'] / 100)
            st.write("Colonna 'importo_offerto' calcolata da importo base e ribasso.")

        # Calcola importo_aggiudicazione se mancano ma ci sono base e ribasso aggiudicatario
        if 'importo_aggiudicazione' not in df.columns and ('importo_base' in df.columns and 'ribasso_aggiudicatario_percentuale' in df.columns):
            mask = df['importo_base'].notna() & df['ribasso_aggiudicatario_percentuale'].notna()
            df.loc[mask, 'importo_aggiudicazione'] = df.loc[mask, 'importo_base'] * (1 - df.loc[mask, 'ribasso_aggiudicatario_percentuale'] / 100)
            st.write("Colonna 'importo_aggiudicazione' calcolata da importo base e ribasso aggiudicatario.")

        # --- Selezione Colonne Finali e Validazione CIG ---
        db_cols_expected = db_utils.EXPECTED_COLUMNS
        # Seleziona solo le colonne che esistono nel DataFrame E sono attese dal DB (escludendo quelle auto-generate)
        final_cols = [col for col in db_cols_expected if col in df.columns and col not in ['id', 'data_inserimento']]
        if not final_cols:
             st.error("Nessuna colonna mappata corrisponde alle colonne attese dal database.")
             return None
        df_final = df[final_cols].copy()

        # Validazione CIG (essenziale)
        if 'identificativo_gara' not in df_final.columns:
            st.error("Colonna 'identificativo_gara' (CIG) MANCANTE dopo la mappatura. Impossibile importare.")
            return None
        elif df_final['identificativo_gara'].astype(str).str.strip().replace('', np.nan).isna().any():
            missing_id_count = df_final['identificativo_gara'].astype(str).str.strip().replace('', np.nan).isna().sum()
            st.warning(f"{missing_id_count} righe non hanno un 'identificativo_gara' (CIG) valido e saranno saltate durante l'importazione.")
            # Opzionale: potresti filtrare qui le righe senza CIG valido
            # df_final = df_final.dropna(subset=['identificativo_gara'])
            # df_final = df_final[df_final['identificativo_gara'].astype(str).str.strip() != '']

        st.success(f"File processato. Colonne pronte per l'importazione: {df_final.columns.tolist()}")
        st.dataframe(df_final.head())
        st.write(f"Numero totale righe pronte (potrebbero esserci CIG duplicati o mancanti): {len(df_final)}")
        return df_final

    except Exception as e:
        st.error(f"Errore imprevisto durante il caricamento/processamento del file: {e}")
        st.error(traceback.format_exc())
        return None

def load_gara_for_editing(gara_id: int):
    """Carica dati gara nello stato sessione per modifica."""
    print(f"Tentativo caricamento dati per modifica Gara ID: {gara_id}")
    # Usa _cache_key_modifier per forzare il ri-caricamento se necessario
    gara_data = db_utils.get_gara_by_id(gara_id, _cache_key_modifier=pd.Timestamp.now())
    if gara_data:
        st.session_state.editing_gara_id = gara_id
        for key in form_widget_keys: # Itera sui widget del form
            db_key = key.replace('widget_', '') # Nome colonna DB
            edit_key = f"edit_{db_key}" # Chiave stato sessione per modifica
            value = gara_data.get(db_key) # Prendi valore dal DB

            # Trasformazioni specifiche per i widget
            if db_key == 'data_gara' and value:
                try:
                    # Converte la stringa data YYYY-MM-DD dal DB in oggetto date per st.date_input
                    value = datetime.datetime.strptime(str(value).split()[0], DATE_FORMAT_STR).date()
                except (ValueError, TypeError):
                    value = None # In caso di errore conversione
            elif db_key == 'posizione_in_graduatoria' and value is None:
                 value = 0 # Rappresenta NULL come 0 nel number_input (come da help text)
            elif pd.isna(value): # Gestione generica di None, NaN, NaT
                value = None

            # Assegna allo stato sessione per modifica, differenziando per tipo di widget
            if db_key in ['importo_base', 'mio_ribasso_percentuale', 'soglia_anomalia_calcolata',
                          'ribasso_aggiudicatario_percentuale', 'numero_concorrenti', 'posizione_in_graduatoria']:
                 # Per number_input, usa None se il valore è None/NaN
                 st.session_state[edit_key] = value if value is not None else None
            elif db_key == 'data_gara':
                 st.session_state[edit_key] = value # Già convertito sopra
            else: # Campi testuali o selectbox
                 # Usa stringa vuota se il valore è None per widget testuali/selectbox
                 st.session_state[edit_key] = value if value is not None else default_form_values.get(key, "")

        print(f"Dati caricati nello stato sessione per modifica Gara ID: {gara_id}")
        st.session_state.edit_form_reset_needed = False # Reset flag perché stiamo caricando NUOVI dati
    else:
        st.error(f"Impossibile caricare i dati per la Gara ID {gara_id}. Potrebbe essere stata eliminata.")
        reset_edit_state() # Resetta lo stato se il caricamento fallisce

def reset_edit_state():
     """Resetta lo stato della sessione relativo alla modifica della gara."""
     st.session_state.editing_gara_id = None
     # Itera sulle chiavi del form di modifica e reimposta ai valori di default
     for key in edit_form_keys:
          # Trova la chiave corrispondente nel form di inserimento per ottenere il default corretto
          edit_base_key = key.replace('edit_', 'widget_')
          # Resetta ai valori di default corretti per tipo (None per numerici/data, "" per stringhe)
          if edit_base_key in ['widget_identificativo_gara', 'widget_descrizione', 'widget_categoria_lavori', 'widget_stazione_appaltante', 'widget_esito', 'widget_note']:
              st.session_state[key] = default_form_values.get(edit_base_key, "")
          else: # Campi numerici e data
              st.session_state[key] = default_form_values.get(edit_base_key, None)
     st.session_state.edit_form_reset_needed = False # Assicura che il flag sia False dopo il reset
     print("Stato sessione di modifica resettato.")


# --- TITOLO ---
st.title("📊 Analisi Gare d'Appalto")
st.caption("Uno strumento per centralizzare e analizzare le tue partecipazioni.")

# --- Check e reset flag stato modifica ---
# Esegui il reset SE il flag è True. Il flag viene impostato a True dopo un update riuscito o annullamento.
if st.session_state.get('edit_form_reset_needed', False):
    print("Reset edit state richiesto dal flag...")
    reset_edit_state()
    # Non serve rerun qui, il rerun che ha triggerato il reset (es. dopo submit) gestirà l'aggiornamento UI

# --- SIDEBAR ---
with st.sidebar:
    st.image("logo.png", width=150) # Esempio logo
    st.header("Azioni Rapide")

    # --- Inserimento Nuova Gara ---
    with st.expander("➕ Inserisci Nuova Gara", expanded=False):
        # Reset campi form se l'inserimento precedente ha avuto successo
        if st.session_state.get('form_submit_success', False):
            print("Resetting insert form fields after successful submission.")
            for key in form_widget_keys: st.session_state[key] = default_form_values[key]
            st.session_state.form_submit_success = False # Resetta il flag

        with st.form("new_gara_form", clear_on_submit=False): # clear_on_submit=False per mantenere i dati in caso di validazione fallita
            st.markdown("**Dettagli Gara**", help="Inserire i dati principali della gara.")
            st.text_input("CIG*", key="widget_identificativo_gara", placeholder="Es: 9876543210", help="Codice Identificativo Gara univoco.")
            st.text_area("Descrizione", key="widget_descrizione", height=100, help="Oggetto della gara.")
            today = datetime.date.today()
            st.date_input("Data Gara*", value=st.session_state.widget_data_gara, min_value=today - datetime.timedelta(days=365*10), max_value=today + datetime.timedelta(days=365*2), key="widget_data_gara", help="Data di scadenza o riferimento della gara.")
            st.number_input("Importo Base (€)", value=st.session_state.widget_importo_base, format=CURRENCY_INTERNAL_FORMAT, step=1000.0, key="widget_importo_base", placeholder="Es: 150000.00", help="Importo a base d'asta senza IVA.")
            st.text_input("Categoria", key="widget_categoria_lavori", placeholder="Es: OG1", help="Categoria prevalente dei lavori.")
            st.text_input("Staz. Appaltante", key="widget_stazione_appaltante", help="Ente che ha bandito la gara.")

            st.markdown("**Offerta**", help="Dati relativi alla tua offerta.")
            st.number_input("Tuo Ribasso (%)*", value=st.session_state.widget_mio_ribasso_percentuale, format=PERCENTAGE_FORMAT, step=0.0001, key="widget_mio_ribasso_percentuale", placeholder="Es: 15.1234", help="Percentuale di ribasso offerta (4 decimali).")
            # Calcolo dinamico importo offerto (solo display)
            importo_offerto_calc_insert = None
            if st.session_state.widget_importo_base is not None and st.session_state.widget_mio_ribasso_percentuale is not None:
                try:
                    importo_offerto_calc_insert = float(st.session_state.widget_importo_base) * (1 - float(st.session_state.widget_mio_ribasso_percentuale) / 100)
                    # Mostra calcolo con formattazione valuta
                    st.caption(f"Importo Offerto Calcolato: {importo_offerto_calc_insert:,.2f} €")
                except (ValueError, TypeError):
                    st.caption("Importo Offerto Calcolato: (dati input non validi)")

            st.markdown("**Esito (opzionale)**", help="Dati sull'esito finale della gara, se noti.")
            st.number_input("Soglia Anom. (%)", value=st.session_state.widget_soglia_anomalia_calcolata, format=PERCENTAGE_FORMAT, step=0.0001, key="widget_soglia_anomalia_calcolata", placeholder="Es: 18.5678", help="Soglia di anomalia calcolata.")
            st.number_input("Ribasso Agg. (%)", value=st.session_state.widget_ribasso_aggiudicatario_percentuale, format=PERCENTAGE_FORMAT, step=0.0001, key="widget_ribasso_aggiudicatario_percentuale", placeholder="Es: 16.9876", help="Ribasso dell'aggiudicatario.")
            st.number_input("Num. Conc.", value=st.session_state.widget_numero_concorrenti, min_value=0, step=1, key="widget_numero_concorrenti", help="Numero di concorrenti totali.")
            st.number_input("Posizione (0 o vuoto se NA)", value=st.session_state.widget_posizione_in_graduatoria, min_value=0, step=1, key="widget_posizione_in_graduatoria", help="Tua posizione finale (0 o vuoto = non in graduatoria/anomala).")
            esito_options = ["", "Aggiudicata", "Persa", "Annullata", "In corso", "Ritirata", "Esclusa (Anomala)"]
            # Gestisce se il valore salvato non è tra le opzioni
            current_esito_insert = st.session_state.widget_esito
            try:
                esito_index = esito_options.index(current_esito_insert) if current_esito_insert in esito_options else 0
            except ValueError: esito_index = 0 # Default a "" se valore non valido
            st.selectbox("Esito", options=esito_options, index=esito_index, key="widget_esito", help="Esito finale della partecipazione.")
            st.text_area("Note", key="widget_note", height=70, help="Eventuali annotazioni aggiuntive.")

            submitted_insert = st.form_submit_button("➕ Salva Nuova Gara", use_container_width=True, type="primary")

        # Logica dopo submit del form di inserimento
        if submitted_insert:
            # Validazione campi obbligatori
            validation_ok = True; error_messages = []
            if not st.session_state.widget_identificativo_gara: error_messages.append("CIG obbligatorio!"); validation_ok = False
            if st.session_state.widget_data_gara is None: error_messages.append("Data Gara obbligatoria!"); validation_ok = False
            if st.session_state.widget_mio_ribasso_percentuale is None: error_messages.append("Tuo Ribasso (%) obbligatorio!"); validation_ok = False
            # Aggiungi altre validazioni se necessario (es: formato CIG, valori numerici sensati)

            if validation_ok:
                # Raccogli dati dallo stato sessione
                gara_data = {k.replace('widget_', ''): v for k, v in st.session_state.items() if k in form_widget_keys}
                # Formatta data per DB
                gara_data['data_gara'] = gara_data['data_gara'].strftime(DATE_FORMAT_STR) if gara_data.get('data_gara') else None
                # Usa importo offerto calcolato
                gara_data['importo_offerto'] = importo_offerto_calc_insert
                # Gestisci posizione 0 come NULL
                if gara_data.get('posizione_in_graduatoria') == 0: gara_data['posizione_in_graduatoria'] = None
                # Pulisci eventuali valori vuoti/None prima di passare al DB (anche se add_gara lo fa già)
                # gara_data_clean = {k: v for k, v in gara_data.items() if not (v is None or (isinstance(v, str) and not v.strip()))}

                if db_utils.add_gara(gara_data): # Passa direttamente gara_data, la pulizia è in add_gara
                    st.success(f"Gara '{gara_data['identificativo_gara']}' salvata con successo!")
                    st.session_state.form_submit_success = True # Flag per resettare i campi al prossimo rerun
                    trigger_data_refresh() # Invalida cache DB
                    st.rerun() # Ricarica la pagina per aggiornare la tabella e resettare il form
                else:
                    # L'errore (es. CIG duplicato) viene mostrato da add_gara tramite st.error
                    st.error(f"Errore durante il salvataggio della Gara '{gara_data['identificativo_gara']}'. Controllare i messaggi di errore sopra.")
            else:
                # Mostra errori di validazione
                for error in error_messages: st.warning(error)

    st.divider()

    # --- Caricamento da File ---
    st.subheader("Carica Dati da File")
    uploaded_file_sb = st.file_uploader("Seleziona file Excel o CSV", type=["csv", "xlsx", "xls"], key="file_uploader_sidebar", label_visibility="collapsed", help="Carica più gare da un file CSV o Excel.")

    # Gestione stato per evitare ricaricamento ad ogni interazione
    if 'df_loaded_sidebar' not in st.session_state: st.session_state.df_loaded_sidebar = None
    if 'uploaded_file_id_sidebar' not in st.session_state: st.session_state.uploaded_file_id_sidebar = None

    if uploaded_file_sb is not None:
        # Identificativo univoco del file caricato
        current_file_id_sb = uploaded_file_sb.id + str(uploaded_file_sb.size) + uploaded_file_sb.name
        # Processa il file solo se è nuovo o non ancora processato
        if st.session_state.df_loaded_sidebar is None or current_file_id_sb != st.session_state.uploaded_file_id_sidebar:
            with st.spinner("Processo file in corso..."):
                st.session_state.df_loaded_sidebar = load_data_from_file(uploaded_file_sb)
            st.session_state.uploaded_file_id_sidebar = current_file_id_sb # Salva ID file processato

        # Se il DataFrame è stato caricato con successo, mostra il bottone di importazione
        if st.session_state.df_loaded_sidebar is not None:
            if st.button("⚡ Importa Dati da File Caricato", key="import_button_sidebar", type="primary", use_container_width=True):
                df_to_import_sb = st.session_state.df_loaded_sidebar
                imported_count_sb = 0
                skipped_count_sb = 0
                error_cigs_sb = []

                with st.spinner("Importazione dati in corso..."):
                    data_to_insert_sb = df_to_import_sb.to_dict('records')
                    for record_sb in data_to_insert_sb:
                        record_cig = record_sb.get('identificativo_gara', 'N/D')
                        # Pulisci record (solo colonne DB valide, gestisci NaN)
                        record_clean_sb = {k: v for k, v in record_sb.items() if pd.notna(v) and k in db_utils.EXPECTED_COLUMNS}
                        # Formatta data se presente e valida
                        if 'data_gara' in record_clean_sb and isinstance(record_clean_sb['data_gara'], (datetime.datetime, pd.Timestamp)):
                             record_clean_sb['data_gara'] = record_clean_sb['data_gara'].strftime(DATE_FORMAT_STR)
                        # Gestisci posizione 0 come NULL
                        if record_clean_sb.get('posizione_in_graduatoria') == 0: record_clean_sb['posizione_in_graduatoria'] = None

                        # Prova ad aggiungere, gestendo CIG mancante/vuoto
                        if 'identificativo_gara' in record_clean_sb and str(record_clean_sb['identificativo_gara']).strip():
                            if db_utils.add_gara(record_clean_sb):
                                imported_count_sb += 1
                            else:
                                skipped_count_sb += 1
                                error_cigs_sb.append(record_cig) # Salva CIG con errore (es. duplicato)
                        else:
                            skipped_count_sb +=1 # Salta righe senza CIG valido
                            error_cigs_sb.append("(CIG Mancante)")

                # Report importazione
                report_msg_sb = f"Importazione completata: {imported_count_sb} gare aggiunte/aggiornate, {skipped_count_sb} saltate/errate."
                st.info(report_msg_sb)
                if error_cigs_sb:
                     st.warning(f"CIG saltati o con errori: {', '.join(list(set(error_cigs_sb))[:10])}{'...' if len(set(error_cigs_sb)) > 10 else ''}") # Mostra alcuni CIG errati

                # Reset stato file upload dopo importazione
                st.session_state.df_loaded_sidebar = None
                st.session_state.uploaded_file_id_sidebar = None
                # Forse resetta anche il widget file_uploader? Non sempre funziona bene
                # st.session_state.file_uploader_sidebar = None
                trigger_data_refresh() # Aggiorna cache DB
                st.rerun() # Ricarica pagina

# --- Area Principale ---
st.header("📚 Storico Gare Inserite")
st.markdown("Visualizza, filtra e gestisci le gare inserite nel database.")

# Carica dati (usa cache di db_utils)
df_gare = db_utils.get_all_gare(_refresh_trigger=st.session_state.refresh_data)

if df_gare.empty:
    st.info("Nessuna gara trovata nel database. Inizia caricando un file o inserendo dati manualmente dalla sidebar.")
else:
    # --- Conversione Tipi Post-Lettura (già fatta in db_utils, ma riassicura per sicurezza) ---
    # Non dovrebbe essere necessario se db_utils.get_all_gare è corretto, ma ridondanza non guasta
    if 'data_gara' in df_gare.columns: df_gare['data_gara'] = pd.to_datetime(df_gare['data_gara'], errors='coerce')
    if 'data_inserimento' in df_gare.columns: df_gare['data_inserimento'] = pd.to_datetime(df_gare['data_inserimento'], errors='coerce')
    numeric_cols_viz = ['importo_base', 'mio_ribasso_percentuale', 'importo_offerto','soglia_anomalia_calcolata', 'ribasso_aggiudicatario_percentuale','importo_aggiudicazione']
    for col in numeric_cols_viz:
        if col in df_gare.columns: df_gare[col] = pd.to_numeric(df_gare[col], errors='coerce')
    integer_cols_viz = ['numero_concorrenti', 'posizione_in_graduatoria']
    for col in integer_cols_viz:
        if col in df_gare.columns: df_gare[col] = pd.to_numeric(df_gare[col], errors='coerce').astype('Int64') # Usa Int64 per gestire NaN in interi

    # --- Filtri ---
    st.subheader("🔍 Filtra Dati Visualizzati")
    # Layout filtri migliorato
    filter_cols = st.columns([1.5, 1, 1, 1.5]) # Proporzioni colonne

    with filter_cols[0]: # Data Gara
        valid_dates = df_gare['data_gara'].dropna()
        date_filter = None
        if not valid_dates.empty:
            min_date_dt = valid_dates.min()
            max_date_dt = valid_dates.max()
            # Conversione a date per il widget
            min_date = min_date_dt.date()
            max_date = max_date_dt.date()

            if min_date != max_date:
                # Default: ultimi 365 giorni o dal min_date se più recente
                default_start_date = max(min_date, max_date - datetime.timedelta(days=365))
                selected_range = st.date_input(
                    "Filtra Data Gara",
                    value=(default_start_date, max_date), # Tuple (start, end)
                    min_value=min_date,
                    max_value=max_date,
                    key="date_filter_range",
                    help="Seleziona un intervallo di date per le gare."
                )
                # Assicura che selected_range sia una tupla di due date
                if len(selected_range) == 2:
                    start_dt = datetime.datetime.combine(selected_range[0], datetime.time.min)
                    end_dt = datetime.datetime.combine(selected_range[1], datetime.time.max)
                    date_filter = (start_dt, end_dt)
                else: # Fallback se l'input non è valido (improbabile con date_input)
                    date_filter = (min_date_dt, max_date_dt)
            else:
                # Caso con una sola data disponibile
                st.markdown(f"**Data Gare:** {min_date.strftime('%d/%m/%Y')}")
                selected_range = (min_date, max_date) # Range fittizio per logica filtro
                start_dt = datetime.datetime.combine(selected_range[0], datetime.time.min)
                end_dt = datetime.datetime.combine(selected_range[1], datetime.time.max)
                date_filter = (start_dt, end_dt)
        else:
            st.caption("Nessuna data gara valida disponibile per filtrare.")

    with filter_cols[1]: # Categoria Lavori
        selected_category = "Tutte"
        # Opzioni: "Tutte" + lista unica e ordinata delle categorie presenti (ignorando NaN)
        cat_options = ["Tutte"] + sorted(df_gare['categoria_lavori'].dropna().astype(str).unique().tolist())
        if len(cat_options) > 1: # Mostra selectbox solo se c'è più di un'opzione oltre a "Tutte"
            selected_category = st.selectbox("Categoria", options=cat_options, key="category_filter", index=0, help="Filtra per categoria lavori.")

    with filter_cols[2]: # Esito
       selected_esito = "Tutti"
       # Opzioni: "Tutti" + lista unica e ordinata degli esiti presenti (ignorando NaN)
       esito_options = ["Tutti"] + sorted(df_gare['esito'].dropna().astype(str).unique().tolist())
       if len(esito_options) > 1:
           selected_esito = st.selectbox("Esito", options=esito_options, key="esito_filter", index=0, help="Filtra per esito gara.")

    with filter_cols[3]: # Importo Base
        selected_importo_range = None
        if 'importo_base' in df_gare.columns and df_gare['importo_base'].notna().any():
             valid_importi = df_gare['importo_base'].dropna()
             min_imp = float(valid_importi.min())
             max_imp = float(valid_importi.max())
             if min_imp < max_imp:
                 # Calcola step dinamico per lo slider
                 step_value = max(1000.0, float(round((max_imp - min_imp) / 100))) # Circa 100 step
                 # Slider per selezionare range importo
                 selected_importo_range = st.slider(
                     "Filtra Importo Base (€)",
                     min_value=min_imp,
                     max_value=max_imp,
                     value=(min_imp, max_imp), # Default: tutto il range
                     step=step_value,
                     format="%.0f €", # Mostra come intero con simbolo €
                     key="importo_filter",
                     help="Seleziona un intervallo per l'importo base d'asta."
                 )
             else: # Caso con un solo valore di importo
                 st.markdown(f"**Importo Base:** {min_imp:,.2f} €")
                 selected_importo_range = (min_imp, max_imp) # Range fittizio

    # --- Applica Filtri ---
    df_filtered = df_gare.copy() # Lavora su una copia per non alterare l'originale
    if date_filter:
        df_filtered = df_filtered.dropna(subset=['data_gara'])
        df_filtered = df_filtered[(df_filtered['data_gara'] >= date_filter[0]) & (df_filtered['data_gara'] <= date_filter[1])]
    if selected_category != "Tutte":
        df_filtered = df_filtered[df_filtered['categoria_lavori'] == selected_category]
    if selected_esito != "Tutti":
        df_filtered = df_filtered[df_filtered['esito'] == selected_esito]
    if selected_importo_range:
        df_filtered = df_filtered.dropna(subset=['importo_base'])
        df_filtered = df_filtered[(df_filtered['importo_base'] >= selected_importo_range[0]) & (df_filtered['importo_base'] <= selected_importo_range[1])]

    # --- Visualizza Tabella Filtrata ---
    st.dataframe( df_filtered,
        hide_index=True,
        use_container_width=True,
        key="main_dataframe",
        column_config={ # Configurazione specifica per colonne
            "id": st.column_config.NumberColumn("ID", width="small", disabled=True),
            "identificativo_gara": st.column_config.TextColumn("CIG", help="Codice Identificativo Gara", width="medium"),
            "descrizione": st.column_config.TextColumn("Descrizione", width="large"),
            "data_gara": st.column_config.DateColumn("Data Gara", format=DATE_DISPLAY_FORMAT), # Usa formato display definito
            "importo_base": st.column_config.NumberColumn("Importo Base", format=CURRENCY_COLUMN_FORMAT, help="Importo base d'asta"), # Usa formato valuta
            "categoria_lavori": st.column_config.TextColumn("Categoria", width="small"),
            "stazione_appaltante": st.column_config.TextColumn("Staz. App.", width="medium"),
            "mio_ribasso_percentuale": st.column_config.NumberColumn("Tuo Rib. (%)", format=PERCENTAGE_DISPLAY_FORMAT, help="Tuo ribasso offerto"), # Usa formato %
            "importo_offerto": st.column_config.NumberColumn("Importo Offerto", format=CURRENCY_COLUMN_FORMAT, help="Importo calcolato della tua offerta"), # Usa formato valuta
            "soglia_anomalia_calcolata": st.column_config.NumberColumn("Soglia Anom. (%)", format=PERCENTAGE_DISPLAY_FORMAT, help="Soglia di anomalia calcolata"), # Usa formato %
            "ribasso_aggiudicatario_percentuale": st.column_config.NumberColumn("Ribasso Agg. (%)", format=PERCENTAGE_DISPLAY_FORMAT, help="Ribasso dell'aggiudicatario"), # Usa formato %
            "importo_aggiudicazione": st.column_config.NumberColumn("Importo Agg.", format=CURRENCY_COLUMN_FORMAT, help="Importo finale di aggiudicazione"), # Usa formato valuta
            "numero_concorrenti": st.column_config.NumberColumn("Num. Conc.", width="small", format="%d", help="Numero totale concorrenti"), # Formato intero
            "posizione_in_graduatoria": st.column_config.NumberColumn("Posiz.", width="small", format="%d", help="Tua posizione in graduatoria (se non esclusa)"), # Formato intero, Int64 gestisce NaN
            "esito": st.column_config.TextColumn("Esito", width="small"),
            "note": st.column_config.TextColumn("Note", width="medium"),
            "data_inserimento": st.column_config.DatetimeColumn("Inserito il", format=DATETIME_FORMAT_STR, disabled=True, width="small"), # Usa formato datetime
        }
    )
    st.write(f"Visualizzate **{len(df_filtered)}** gare su **{len(df_gare)}** totali nel database (in base ai filtri applicati).")

    # --- Download CSV ---
    @st.cache_data # Cache conversione CSV
    def convert_df_to_csv(df_to_convert):
        """Converte DataFrame in CSV con encoding e separatori specifici."""
        try:
            # Usa separatore ';' e decimale ',' comuni in Italia, encoding con BOM per Excel
            return df_to_convert.to_csv(index=False, sep=';', decimal=',', date_format=DATE_FORMAT_STR, encoding='utf-8-sig').encode('utf-8-sig')
        except Exception as e_csv:
            st.error(f"Errore durante la conversione in CSV: {e_csv}")
            return None

    csv_data = convert_df_to_csv(df_filtered)
    if csv_data:
        st.download_button(
            label="📥 Scarica Dati Filtrati (CSV)",
            data=csv_data,
            file_name=f"gare_filtrate_{datetime.date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Scarica la tabella attualmente visualizzata in formato CSV (compatibile Excel Italia)."
        )
    st.divider()

    # --- Sezione Azioni: Modifica e Elimina ---
    st.subheader("✍️ Azioni sulle Gare Filtrate")
    st.caption("Seleziona una gara dalla lista sottostante per modificarne i dettagli o eliminarla.")
    col_actions_1, col_actions_2 = st.columns(2)

    with col_actions_1: # Modifica Gara
        st.markdown("**Modifica Gara:**")
        if not df_filtered.empty:
            # Crea opzioni per selectbox: "ID - CIG (Data)" -> ID
            edit_options = {
                f"{int(row['id'])} - {row['identificativo_gara']} ({row['data_gara'].strftime('%d/%m/%Y') if pd.notna(row['data_gara']) else 'N/D'})": int(row['id'])
                for _, row in df_filtered.sort_values('data_gara', ascending=False).iterrows() # Ordina per data decrescente nel selectbox
            }
            edit_options_list = ["--- Seleziona per MODIFICARE ---"] + list(edit_options.keys())

            # Trova l'indice corrente basato sull'ID in modifica (se presente)
            current_edit_selection_index = 0
            if st.session_state.editing_gara_id is not None:
                # Cerca la stringa corrispondente all'ID in modifica
                for i, key_str in enumerate(edit_options_list):
                    if key_str != edit_options_list[0] and edit_options.get(key_str) == st.session_state.editing_gara_id:
                        current_edit_selection_index = i
                        break

            # Selectbox per scegliere la gara da modificare
            selected_gara_str_edit = st.selectbox(
                "Seleziona Gara da Modificare:",
                options=edit_options_list,
                index=current_edit_selection_index,
                key="gara_to_edit_select",
                label_visibility="collapsed",
                help="Scegli la gara da modificare dall'elenco."
            )

            selected_id_edit = edit_options.get(selected_gara_str_edit) # Ottiene l'ID numerico o None

            # Logica di caricamento/reset basata sulla selezione
            if selected_id_edit is not None: # Un ID valido è stato selezionato
                # Carica i dati solo se l'ID selezionato è DIVERSO da quello già in modifica
                # Questo evita loop di caricamento/rerun se l'utente non cambia selezione
                if selected_id_edit != st.session_state.editing_gara_id:
                    print(f"Selezione modifica cambiata a ID: {selected_id_edit}")
                    load_gara_for_editing(selected_id_edit)
                    st.rerun() # Ricarica per mostrare il form di modifica aggiornato
            elif st.session_state.editing_gara_id is not None and selected_gara_str_edit == edit_options_list[0]:
                 # L'utente è tornato all'opzione placeholder => resetta lo stato di modifica
                 print("Placeholder selezionato per modifica, resetto stato.")
                 # reset_edit_state() # Il flag farà il reset al prossimo giro
                 st.session_state.edit_form_reset_needed = True
                 st.rerun() # Ricarica per nascondere il form
        else:
            st.caption("Nessuna gara disponibile nei risultati filtrati per la modifica.")

    with col_actions_2: # Eliminazione Gara
            st.markdown("**Elimina Gara:**")
            if not df_filtered.empty:
                # Crea opzioni per selectbox: "ID - CIG (Data)" -> ID
                delete_options = {
                    f"{int(row['id'])} - {row['identificativo_gara']} ({row['data_gara'].strftime('%d/%m/%Y') if pd.notna(row['data_gara']) else 'N/D'})": int(row['id'])
                    for _, row in df_filtered.sort_values('data_gara', ascending=False).iterrows()
                }
                delete_options_list = ["--- Seleziona per ELIMINARE ---"] + list(delete_options.keys())

                # Inizializza lo stato sessione solo se non esiste
                # Questo è il modo corretto per impostare un valore iniziale
                if 'selected_gara_to_delete' not in st.session_state:
                    st.session_state.selected_gara_to_delete = delete_options_list[0]

                # Determina l'indice iniziale basato sullo stato sessione
                # Assicurati che il valore nello stato esista ancora nelle opzioni
                current_selection_value = st.session_state.selected_gara_to_delete
                if current_selection_value not in delete_options_list:
                    current_selection_value = delete_options_list[0] # Fallback al placeholder se il valore salvato non è più valido
                current_index = delete_options_list.index(current_selection_value)

                # Crea il widget selectbox. Il suo valore sarà nello stato sessione con la chiave specificata
                selected_gara_str_delete = st.selectbox(
                    "Seleziona Gara da Eliminare:",
                    options=delete_options_list,
                    key="selected_gara_to_delete", # La chiave è usata da Streamlit per gestire lo stato
                    index=current_index,           # Imposta la selezione iniziale
                    label_visibility="collapsed",
                    help="Scegli la gara da eliminare dall'elenco."
                )

                # !!! RIMUOVI QUESTA RIGA !!!
                # st.session_state.selected_gara_to_delete = selected_gara_str_delete

                # Usa il valore restituito da st.selectbox per la logica successiva
                if selected_gara_str_delete != delete_options_list[0]:
                    gara_id_to_delete = delete_options[selected_gara_str_delete]
                    gara_cig_to_delete = selected_gara_str_delete.split(' - ')[1].split(' (')[0] # Estrai CIG per messaggio

                    # Messaggio di conferma e bottoni
                    st.warning(f"Sei sicuro di voler eliminare la Gara ID: **{gara_id_to_delete}** (CIG: {gara_cig_to_delete})? L'azione è irreversibile.")
                    col_del1, col_del2 = st.columns([1, 1]) # Colonne per bottoni conferma/annulla
                    with col_del1:
                        if st.button("🔴 CONFERMA ELIMINA", key=f"delete_confirm_{gara_id_to_delete}", type="primary", use_container_width=True):
                            with st.spinner(f"Eliminazione Gara ID {gara_id_to_delete}..."):
                                if db_utils.delete_gara_by_id(gara_id_to_delete):
                                    st.success(f"Gara ID {gara_id_to_delete} eliminata con successo.")
                                    # Resetta il VALORE del widget nello stato sessione al placeholder DOPO l'azione
                                    st.session_state.selected_gara_to_delete = delete_options_list[0]
                                    if st.session_state.editing_gara_id == gara_id_to_delete:
                                        st.session_state.edit_form_reset_needed = True
                                    trigger_data_refresh()
                                    st.rerun()
                                else:
                                    st.error(f"Errore durante l'eliminazione della Gara ID {gara_id_to_delete}.")
                    with col_del2:
                        if st.button("Annulla", key=f"delete_cancel_{gara_id_to_delete}", use_container_width=True):
                            # Resetta il VALORE del widget nello stato sessione al placeholder
                            st.session_state.selected_gara_to_delete = delete_options_list[0]
                            st.rerun()
            else:
                 st.caption("Nessuna gara disponibile nei risultati filtrati per l'eliminazione.")


    # --- Form di Modifica (visibile solo se editing_gara_id è impostato) ---
    if st.session_state.editing_gara_id is not None:
        st.divider()
        st.subheader(f"📝 Modifica Dati Gara ID: {st.session_state.editing_gara_id} (CIG: {st.session_state.get('edit_identificativo_gara', 'N/D')})")

        with st.form("edit_gara_form", clear_on_submit=False):
            c1_edit, c2_edit = st.columns(2)
            # Colonna Sinistra: Dettagli Gara
            with c1_edit:
                st.markdown("**Dettagli Gara**")
                st.text_input("CIG (Non modificabile)", value=st.session_state.get('edit_identificativo_gara'), disabled=True)
                st.text_area("Descrizione", key="edit_descrizione", height=100)
                today_edit = datetime.date.today()
                st.date_input("Data Gara*", value=st.session_state.get('edit_data_gara'),
                              min_value=today_edit - datetime.timedelta(days=365*10),
                              max_value=today_edit + datetime.timedelta(days=365*2), key="edit_data_gara")
                st.number_input("Importo Base (€)", value=st.session_state.get('edit_importo_base'), format=CURRENCY_INTERNAL_FORMAT, step=1000.0, key="edit_importo_base")
                st.text_input("Categoria", key="edit_categoria_lavori")
                st.text_input("Staz. App.", key="edit_stazione_appaltante")

            # Colonna Destra: Offerta ed Esito
            with c2_edit:
                st.markdown("**Offerta**")
                st.number_input("Tuo Ribasso (%)*", value=st.session_state.get('edit_mio_ribasso_percentuale'), format=PERCENTAGE_FORMAT, step=0.0001, key="edit_mio_ribasso_percentuale")
                # Calcolo dinamico importo offerto (display)
                importo_offerto_calc_edit = None
                edit_imp_base = st.session_state.get('edit_importo_base')
                edit_mio_rib = st.session_state.get('edit_mio_ribasso_percentuale')
                if edit_imp_base is not None and edit_mio_rib is not None:
                    try:
                        importo_offerto_calc_edit = float(edit_imp_base) * (1 - float(edit_mio_rib) / 100)
                        st.caption(f"Importo Offerto Calcolato: {importo_offerto_calc_edit:,.2f} €")
                    except (ValueError, TypeError):
                        st.caption("Importo Offerto Calcolato: (dati input non validi)")

                st.markdown("**Esito**")
                st.number_input("Soglia Anom. (%)", value=st.session_state.get('edit_soglia_anomalia_calcolata'), format=PERCENTAGE_FORMAT, step=0.0001, key="edit_soglia_anomalia_calcolata")
                st.number_input("Ribasso Agg. (%)", value=st.session_state.get('edit_ribasso_aggiudicatario_percentuale'), format=PERCENTAGE_FORMAT, step=0.0001, key="edit_ribasso_aggiudicatario_percentuale")
                st.number_input("Num. Conc.", value=st.session_state.get('edit_numero_concorrenti'), min_value=0, step=1, key="edit_numero_concorrenti")
                # Posizione: Ricorda che 0 qui significa NA/Anomala
                st.number_input("Posizione (0 o vuoto se NA)", value=st.session_state.get('edit_posizione_in_graduatoria', 0), min_value=0, step=1, key="edit_posizione_in_graduatoria") # Default a 0 se non presente
                esito_options_edit = ["", "Aggiudicata", "Persa", "Annullata", "In corso", "Ritirata", "Esclusa (Anomala)"]
                current_esito_edit = st.session_state.get('edit_esito', "")
                try:
                    esito_index_edit = esito_options_edit.index(current_esito_edit) if current_esito_edit in esito_options_edit else 0
                except ValueError: esito_index_edit = 0
                st.selectbox("Esito", options=esito_options_edit, index=esito_index_edit, key="edit_esito")
                st.text_area("Note", key="edit_note", height=70)

            # Bottoni Submit e Annulla DENTRO il form
            submit_col, cancel_col = st.columns(2)
            with submit_col:
                submitted_edit = st.form_submit_button("💾 Salva Modifiche", type="primary", use_container_width=True)
            with cancel_col:
                # Il bottone Annulla qui è anch'esso un 'submit' del form, ma con logica diversa sotto
                cancelled_edit = st.form_submit_button("❌ Annulla Modifica", use_container_width=True)

        # Logica DOPO la pressione di uno dei bottoni del form di modifica
        if submitted_edit:
            # Validazione campi obbligatori
            edit_validation_ok = True; edit_error_messages = []
            if st.session_state.edit_data_gara is None: edit_error_messages.append("Data Gara obbligatoria!"); edit_validation_ok = False
            if st.session_state.edit_mio_ribasso_percentuale is None: edit_error_messages.append("Tuo Ribasso (%) obbligatorio!"); edit_validation_ok = False

            if edit_validation_ok:
                # Raccogli dati correnti dallo stato 'edit_*'
                gara_data_update = {k.replace('edit_', ''): st.session_state[k] for k in edit_form_keys if k in st.session_state}
                gara_data_update['data_gara'] = gara_data_update['data_gara'].strftime(DATE_FORMAT_STR) if gara_data_update.get('data_gara') else None
                gara_data_update['importo_offerto'] = importo_offerto_calc_edit
                # Gestisci posizione 0 come NULL per il DB
                if gara_data_update.get('posizione_in_graduatoria') == 0: gara_data_update['posizione_in_graduatoria'] = None

                # Prepara dati per l'update (la funzione update_gara gestirà la pulizia finale)
                current_editing_id = st.session_state.editing_gara_id
                if db_utils.update_gara(current_editing_id, gara_data_update):
                    st.success(f"Gara ID {current_editing_id} aggiornata con successo!")
                    st.session_state.edit_form_reset_needed = True # Imposta flag per reset al prox rerun
                    trigger_data_refresh() # Aggiorna cache DB
                    st.rerun() # Ricarica pagina
                else:
                    # L'errore viene mostrato da update_gara
                    st.error(f"Errore durante l'aggiornamento della Gara ID {current_editing_id}.")
            else:
                # Mostra errori di validazione (idealmente DENTRO il form, ma qui va bene sopra/sotto)
                for error in edit_error_messages: st.warning(error)

        if cancelled_edit:
             print("Annulla Modifica premuto.")
             st.session_state.edit_form_reset_needed = True # Flag per reset al prossimo giro
             st.rerun() # Triggera il rerun per far scattare il reset

    st.divider()

    # --- Dashboard e Analisi (Solo se ci sono dati filtrati) ---
    if not df_filtered.empty:
        st.header("📈 Dashboard Analitica")
        st.markdown("Metriche e grafici calcolati sui dati **filtrati** visualizzati nella tabella sopra.")

        # --- KPI Principali ---
        total_gare_filtrate = len(df_filtered)
        gare_vinte_df = df_filtered[df_filtered['esito'] == 'Aggiudicata']
        gare_vinte = len(gare_vinte_df)
        # Considera solo gare con esito definito per calcolare Win Rate sensato
        gare_partecipate_con_esito = df_filtered[df_filtered['esito'].isin(['Aggiudicata', 'Persa'])].shape[0]
        win_rate = (gare_vinte / gare_partecipate_con_esito * 100) if gare_partecipate_con_esito > 0 else 0
        # Calcola medie solo su valori non NaN
        ribasso_medio_tuo = df_filtered['mio_ribasso_percentuale'].mean()
        ribasso_medio_agg = df_filtered['ribasso_aggiudicatario_percentuale'].mean()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("N. Gare Filtrate", total_gare_filtrate, help="Numero totale di gare corrispondenti ai filtri applicati.")
        kpi2.metric(f"Gare Vinte ({gare_partecipate_con_esito})", f"{gare_vinte} ({win_rate:.1f}%)", help=f"Numero di gare aggiudicate rispetto al totale con esito 'Aggiudicata' o 'Persa' ({gare_partecipate_con_esito}).")
        kpi3.metric("Tuo Ribasso Medio", PERCENTAGE_METRIC_FORMAT.format(ribasso_medio_tuo) if pd.notna(ribasso_medio_tuo) else "N/D", help="Media percentuale dei tuoi ribassi offerti (sui dati filtrati).")
        kpi4.metric("Ribasso Medio Agg.", PERCENTAGE_METRIC_FORMAT.format(ribasso_medio_agg) if pd.notna(ribasso_medio_agg) else "N/D", help="Media percentuale dei ribassi degli aggiudicatari (sui dati filtrati).")
        st.divider()


        # --- Analisi Delta Ribassi ---
        st.subheader("📉 Analisi Delta Ribassi", help="Differenza tra il tuo ribasso e quello dell'aggiudicatario (per gare perse) o la soglia di anomalia.")
        df_analysis = df_filtered.copy() # Usa copia per calcoli

        # Calcola delta solo se le colonne necessarie esistono e sono numeriche
        delta_vs_agg_col = 'delta_vs_aggiudicatario'
        delta_vs_soglia_col = 'delta_vs_soglia'
        valid_delta_agg = False
        valid_delta_soglia = False

        if 'mio_ribasso_percentuale' in df_analysis.columns and 'ribasso_aggiudicatario_percentuale' in df_analysis.columns:
            # Assicura tipi numerici prima della sottrazione
            mio_rib_num = pd.to_numeric(df_analysis['mio_ribasso_percentuale'], errors='coerce')
            agg_rib_num = pd.to_numeric(df_analysis['ribasso_aggiudicatario_percentuale'], errors='coerce')
            df_analysis[delta_vs_agg_col] = mio_rib_num - agg_rib_num
            valid_delta_agg = True

        if 'mio_ribasso_percentuale' in df_analysis.columns and 'soglia_anomalia_calcolata' in df_analysis.columns:
            mio_rib_num = pd.to_numeric(df_analysis['mio_ribasso_percentuale'], errors='coerce')
            soglia_num = pd.to_numeric(df_analysis['soglia_anomalia_calcolata'], errors='coerce')
            df_analysis[delta_vs_soglia_col] = mio_rib_num - soglia_num
            valid_delta_soglia = True

        delta_col1, delta_col2 = st.columns(2)
        with delta_col1: # Delta vs Aggiudicatario (per gare perse)
            if valid_delta_agg:
                # Filtra per gare PERSE e dove il delta è calcolabile (non NaN)
                gare_perse_con_delta = df_analysis.loc[df_analysis['esito'] == 'Persa'].dropna(subset=[delta_vs_agg_col])
                if not gare_perse_con_delta.empty:
                    avg_delta_agg = gare_perse_con_delta[delta_vs_agg_col].mean()
                    st.metric("Delta Medio vs Agg. (Perse)",
                              PERCENTAGE_METRIC_FORMAT.format(avg_delta_agg) if pd.notna(avg_delta_agg) else "N/D",
                              help="Media (Tuo Ribasso % - Ribasso Agg. %) solo per gare perse con dati disponibili. Negativo = hai offerto di meno.")
                    st.caption(f"Basato su {len(gare_perse_con_delta)} gare perse.")
                else:
                    st.caption("Nessuna gara 'Persa' con dati sufficienti per calcolare il Delta vs Aggiudicatario.")
            else:
                st.caption("Colonne mancanti per calcolare Delta vs Aggiudicatario.")

        with delta_col2: # Delta vs Soglia (per tutte le gare con dati)
            if valid_delta_soglia:
                # Filtra solo per gare dove il delta vs soglia è calcolabile (non NaN)
                gare_con_delta_soglia = df_analysis.dropna(subset=[delta_vs_soglia_col])
                if not gare_con_delta_soglia.empty:
                    avg_delta_soglia = gare_con_delta_soglia[delta_vs_soglia_col].mean()
                    st.metric("Delta Medio vs Soglia",
                              PERCENTAGE_METRIC_FORMAT.format(avg_delta_soglia) if pd.notna(avg_delta_soglia) else "N/D",
                              help="Media (Tuo Ribasso % - Soglia Anomalia %) per tutte le gare con dati disponibili. Negativo = sei sotto soglia.")
                    st.caption(f"Basato su {len(gare_con_delta_soglia)} gare con soglia.")
                else:
                    st.caption("Nessuna gara con dati sufficienti per calcolare il Delta vs Soglia.")
            else:
                st.caption("Colonne mancanti per calcolare Delta vs Soglia.")

        # Grafico distribuzione delta
        delta_cols_to_plot = [col for col, valid in [(delta_vs_agg_col, valid_delta_agg), (delta_vs_soglia_col, valid_delta_soglia)] if valid]
        if delta_cols_to_plot:
            # Prepara dati per grafico: melt e rimuovi NaN
            df_deltas_long = df_analysis.melt(
                id_vars=['id'], # Mantieni ID o altra chiave univoca se serve
                value_vars=delta_cols_to_plot,
                var_name='Tipo Delta',
                value_name='Valore Delta (%)'
            )
            df_deltas_long.dropna(subset=['Valore Delta (%)'], inplace=True)

            if not df_deltas_long.empty:
                 # Rinomina per legenda più chiara
                 delta_rename_map = {
                     delta_vs_agg_col: 'Delta vs Agg. (Tuo - Agg.)',
                     delta_vs_soglia_col: 'Delta vs Soglia (Tuo - Soglia)'
                 }
                 df_deltas_long['Tipo Delta'] = df_deltas_long['Tipo Delta'].map(delta_rename_map)

                 # Istogramma sovrapposto
                 fig_delta_hist = px.histogram(df_deltas_long,
                                                x='Valore Delta (%)',
                                                color='Tipo Delta', # Colora per tipo di delta
                                                barmode='overlay', # Sovrapponi le barre
                                                title="Distribuzione Delta Ribassi (%)",
                                                nbins=30, # Numero di bin per l'istogramma
                                                opacity=0.7, # Trasparenza per vedere sovrapposizioni
                                                histnorm='percent', # Mostra percentuale invece di conteggio assoluto
                                                labels={'Valore Delta (%)': 'Differenza Percentuale (%)'}
                                                )
                 fig_delta_hist.update_layout(
                     xaxis_ticksuffix="%", # Aggiunge '%' all'asse X
                     yaxis_title="Percentuale Gare", # Etichetta asse Y
                     legend_title_text='Tipo di Delta' # Titolo legenda
                 )
                 st.plotly_chart(fig_delta_hist, use_container_width=True)
                 st.caption("Un delta positivo indica che il tuo ribasso era più alto (meno conveniente per la SA) del valore di confronto (aggiudicatario o soglia).")
            else:
                st.caption("Nessun dato Delta valido disponibile per il grafico di distribuzione.")
        st.divider()


        # --- Grafici Generali ---
        st.subheader("📊 Grafici Generali")
        graph_col1, graph_col2 = st.columns(2)

        with graph_col1: # Grafico Temporale
            st.markdown("**Andamento % nel Tempo**", help="Evoluzione dei ribassi e della soglia nel tempo (sui dati filtrati).")
            df_plot_time = df_filtered.sort_values('data_gara').dropna(subset=['data_gara'])
            # Colonne da plottare sull'asse Y
            plot_cols_time = []
            hover_data_time = ['identificativo_gara', 'importo_base', 'esito'] # Info extra nel tooltip
            # Includi colonne solo se esistono e hanno dati validi
            for col in ['mio_ribasso_percentuale', 'ribasso_aggiudicatario_percentuale', 'soglia_anomalia_calcolata']:
                 if col in df_plot_time.columns and df_plot_time[col].notna().any():
                      plot_cols_time.append(col)
            # Assicura che le colonne hover esistano
            hover_data_time = [col for col in hover_data_time if col in df_plot_time.columns]

            if not df_plot_time.empty and plot_cols_time:
                try:
                    # Prepara i dati per Plotly Express (formato 'long')
                    df_melted = df_plot_time.melt(id_vars=['data_gara'] + hover_data_time,
                                                  value_vars=plot_cols_time,
                                                  var_name='Tipo Percentuale',
                                                  value_name='Valore (%)')
                    # Crea il grafico a linee
                    fig_time = px.line(df_melted, x='data_gara', y='Valore (%)', color='Tipo Percentuale',
                                       title="Ribassi e Soglia nel Tempo",
                                       labels={'Valore (%)': '%', 'data_gara': 'Data Gara', 'Tipo Percentuale': 'Tipo'},
                                       markers=True, # Mostra punti sui dati
                                       hover_data=hover_data_time) # Aggiungi dati hover
                    # Formattazione assi e tooltip
                    fig_time.update_layout(yaxis_tickformat=PERCENTAGE_FORMAT, yaxis_ticksuffix="%")
                    fig_time.update_traces(
                        hovertemplate='<b>Data</b>: %{x|%d/%m/%Y}<br><b>Valore</b>: %{y:'+PERCENTAGE_FORMAT+'}%<br><b>Tipo</b>: %{fullData.name}<br>'+ # Usa fullData.name per il nome corretto della traccia
                                      '<b>CIG</b>: %{customdata[0]}<br>'+
                                      '<b>Importo</b>: %{customdata[1]:,.2f} €<br>'+ # Formatta valuta in hover
                                      '<b>Esito</b>: %{customdata[2]}<extra></extra>' # <extra> rimuove info traccia extra
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                except Exception as e_time:
                    st.warning(f"Errore durante la creazione del grafico temporale: {e_time}")
                    # print(traceback.format_exc()) # Per debug
            else:
                st.caption("Dati insufficienti per generare il grafico temporale (controlla filtri e presenza di date/percentuali).")

        with graph_col2: # Istogramma Tuo Ribasso
            st.markdown("**Distribuzione Tuo Ribasso (%)**", help="Frequenza dei tuoi ribassi offerti (sui dati filtrati).")
            if 'mio_ribasso_percentuale' in df_filtered.columns and df_filtered['mio_ribasso_percentuale'].notna().any():
                fig_hist = px.histogram(df_filtered.dropna(subset=['mio_ribasso_percentuale']),
                                        x='mio_ribasso_percentuale',
                                        nbins=20, # Numero di barre
                                        title="Distribuzione Tuoi Ribassi Offerti")
                fig_hist.update_layout(
                    xaxis_title="Tuo Ribasso Offerto (%)",
                    yaxis_title="Numero Gare",
                    xaxis_tickformat=PERCENTAGE_FORMAT, # Formato asse X
                    xaxis_ticksuffix="%" # Simbolo % asse X
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.caption("Dati 'mio_ribasso_percentuale' insufficienti per generare l'istogramma.")
        st.divider()


        # --- Grafico Scatter: Posizionamento vs Soglia ---
        st.subheader("🎯 Posizionamento Offerta vs Soglia", help="Visualizza il tuo ribasso rispetto alla soglia di anomalia. La dimensione del punto indica l'importo base.")
        required_cols_scatter = ['mio_ribasso_percentuale', 'soglia_anomalia_calcolata']
        if all(col in df_filtered.columns for col in required_cols_scatter):
            # Prepara dati: rimuovi NaN per le colonne essenziali
            df_scatter = df_filtered.dropna(subset=required_cols_scatter).copy()
            if not df_scatter.empty:
                # Aggiungi colonna calcolata per posizione relativa
                df_scatter['Posizione vs Soglia'] = df_scatter.apply(
                    lambda row: 'Sopra Soglia (> Anomalo)' if row['mio_ribasso_percentuale'] > row['soglia_anomalia_calcolata'] else 'Sotto/Uguale Soglia (<= Non Anomalo)',
                    axis=1
                )
                # Aggiungi delta per tooltip (se calcolato prima)
                if valid_delta_soglia and delta_vs_soglia_col in df_analysis.columns:
                     # Assicurati che gli indici corrispondano se df_analysis è stato filtrato diversamente
                     df_scatter = df_scatter.join(df_analysis[delta_vs_soglia_col].rename('Delta da Soglia (%)'), how='left')
                else:
                     df_scatter['Delta da Soglia (%)'] = np.nan # Colonna vuota se non calcolabile

                # Colonne da mostrare nel tooltip
                hover_data_scatter_cols = ['identificativo_gara', 'data_gara', 'Delta da Soglia (%)', 'importo_base', 'ribasso_aggiudicatario_percentuale', 'esito']
                valid_hover_cols = [col for col in hover_data_scatter_cols if col in df_scatter.columns]
                hover_data_scatter = df_scatter[valid_hover_cols] if valid_hover_cols else None

                # Crea grafico scatter
                fig_scatter = px.scatter(
                    df_scatter,
                    x='soglia_anomalia_calcolata',
                    y='mio_ribasso_percentuale',
                    color='esito' if 'esito' in df_scatter.columns else None, # Colora per esito (se disponibile)
                    symbol='Posizione vs Soglia', # Simbolo diverso sopra/sotto soglia
                    size='importo_base' if 'importo_base' in df_scatter.columns and df_scatter['importo_base'].notna().any() else None, # Dimensione per importo (se disponibile)
                    custom_data=hover_data_scatter, # Dati per tooltip personalizzato
                    title="Tuo Ribasso vs Soglia di Anomalia",
                    labels={'mio_ribasso_percentuale': 'Tuo Ribasso Offerto (%)',
                            'soglia_anomalia_calcolata': 'Soglia Anomalia Calcolata (%)',
                            'Posizione vs Soglia': 'Posizionamento Relativo'},
                    color_discrete_sequence=px.colors.qualitative.Plotly # Palette colori
                )

                # Aggiungi linea y=x (bisettrice) per riferimento visivo
                min_val = min(df_scatter['soglia_anomalia_calcolata'].min(), df_scatter['mio_ribasso_percentuale'].min())
                max_val = max(df_scatter['soglia_anomalia_calcolata'].max(), df_scatter['mio_ribasso_percentuale'].max())
                fig_scatter.add_shape(type="line", line=dict(dash='dash', color='grey'),
                                      x0=min_val, y0=min_val, x1=max_val, y1=max_val)

                # Formattazione assi e tooltip
                fig_scatter.update_layout(
                    xaxis_tickformat=PERCENTAGE_FORMAT, xaxis_ticksuffix="%",
                    yaxis_tickformat=PERCENTAGE_FORMAT, yaxis_ticksuffix="%"
                )
                if hover_data_scatter is not None:
                    # Costruisci hovertemplate dinamico
                    ht = "<b>Soglia</b>: %{x:" + PERCENTAGE_FORMAT + "}%<br>"
                    ht += "<b>Tuo Ribasso</b>: %{y:" + PERCENTAGE_FORMAT + "}%<br>"
                    ht += "<b>Esito</b>: %{color}<br>" # Se color è 'esito'
                    ht += "<b>Posizione</b>: %{marker.symbol}<br>" # Mostra il nome del simbolo
                    if 'importo_base' in valid_hover_cols: ht += f"<b>Importo Base</b>: %{{customdata[{valid_hover_cols.index('importo_base')}]:,.2f}} €<br>"
                    if 'identificativo_gara' in valid_hover_cols: ht += f"<b>CIG</b>: %{{customdata[{valid_hover_cols.index('identificativo_gara')}]}}<br>"
                    if 'data_gara' in valid_hover_cols: ht += f"<b>Data</b>: %{{customdata[{valid_hover_cols.index('data_gara')}]|%d/%m/%Y}}<br>" # Formatta data in hover
                    if 'Delta da Soglia (%)' in valid_hover_cols: ht += f"<b>Delta da Soglia</b>: %{{customdata[{valid_hover_cols.index('Delta da Soglia (%)')}]:.4f}}%<br>"
                    if 'ribasso_aggiudicatario_percentuale' in valid_hover_cols: ht += f"<b>Ribasso Agg.</b>: %{{customdata[{valid_hover_cols.index('ribasso_aggiudicatario_percentuale')}]:.4f}}%<br>"
                    ht += "<extra></extra>"
                    fig_scatter.update_traces(hovertemplate=ht)

                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.caption("Nessuna gara con dati sufficienti (tuo ribasso E soglia) per generare il grafico di posizionamento.")
        else:
            st.caption("Colonne 'mio_ribasso_percentuale' e/o 'soglia_anomalia_calcolata' mancanti nei dati.")
        st.divider()


        # --- Analisi per Segmenti ---
        st.subheader("🧩 Analisi per Segmenti", help="Analizza le performance aggregate per Categoria Lavori o per Fascia d'Importo Base.")
        # Scelta tipo segmentazione
        segment_type = st.radio("Raggruppa Dati Per:", ["Categoria Lavori", "Fascia Importo"], horizontal=True, key="segment_radio", index=0)

        df_segmented = df_filtered.copy()
        segment_col = 'segmento' # Nome colonna standard per il raggruppamento
        segment_valid = False # Flag per indicare se la segmentazione è possibile

        # Logica segmentazione per Categoria
        if segment_type == "Categoria Lavori":
            if 'categoria_lavori' in df_segmented.columns and df_segmented['categoria_lavori'].notna().any():
                # Riempi NaN con "Non Specificata" e assicurati sia stringa
                df_segmented[segment_col] = df_segmented['categoria_lavori'].fillna("Non Specificata").astype(str)
                segment_valid = True
            else:
                st.warning("Colonna 'categoria_lavori' non disponibile o vuota nei dati filtrati per segmentare.")

        # Logica segmentazione per Fascia d'Importo
        else: # Fascia Importo
            if 'importo_base' in df_segmented.columns and df_segmented['importo_base'].notna().any():
                df_importo_valid = df_segmented.dropna(subset=['importo_base'])
                if not df_importo_valid.empty:
                    min_imp_seg = df_importo_valid['importo_base'].min()
                    max_imp_seg = df_importo_valid['importo_base'].max()
                    # Definisci fasce (bins) dinamicamente o staticamente
                    # Esempio fasce statiche (adattabili)
                    bins = [-np.inf, 50000, 150000, 500000, 1000000, 5000000, np.inf]
                    labels = ["<50k", "50k-150k", "150k-500k", "500k-1M", "1M-5M", ">5M"]
                    # Applica pd.cut per creare la colonna segmento
                    df_segmented[segment_col] = pd.cut(df_segmented['importo_base'], bins=bins, labels=labels, right=False) # right=False include il limite inferiore
                    segment_valid = True
                else:
                     st.warning("Nessun importo base valido nei dati filtrati per segmentare per fascia.")
            else:
                st.warning("Colonna 'importo_base' non disponibile o vuota nei dati filtrati per segmentare per fascia.")

        # Calcola e visualizza statistiche se la segmentazione è valida
        if segment_valid:
            try:
                # Funzioni di aggregazione desiderate
                agg_funcs = {
                    'id': 'size', # Conteggio gare totali per segmento
                    'mio_ribasso_percentuale': 'mean',
                    'ribasso_aggiudicatario_percentuale': 'mean',
                    'soglia_anomalia_calcolata': 'mean',
                    'numero_concorrenti': 'mean' # Esempio: aggiungi media concorrenti
                }
                # Raggruppa e aggrega
                # observed=False include tutte le categorie (anche quelle vuote nei dati filtrati, se create da pd.cut)
                segment_stats = df_segmented.groupby(segment_col, observed=False).agg(agg_funcs).rename(columns={'id': 'Num. Gare'})

                # Calcola separatamente Vinte e Partecipate (con esito A/P) per Win Rate
                segment_wins = df_segmented[df_segmented['esito'] == 'Aggiudicata'].groupby(segment_col, observed=False).size().rename("Vinte")
                segment_partecipate = df_segmented[df_segmented['esito'].isin(['Aggiudicata', 'Persa'])].groupby(segment_col, observed=False).size().rename("Partecipate (A/P)")

                # Unisci i conteggi Vinte/Partecipate alle statistiche principali
                segment_stats = segment_stats.join(segment_wins, how='left').join(segment_partecipate, how='left').fillna(0)
                # Assicura che Vinte/Partecipate siano interi
                segment_stats[['Vinte', 'Partecipate (A/P)']] = segment_stats[['Vinte', 'Partecipate (A/P)']].astype(int)

                # Calcola Win Rate (%)
                segment_stats['Win Rate (%)'] = ((segment_stats['Vinte'] / segment_stats['Partecipate (A/P)']) * 100).where(segment_stats['Partecipate (A/P)'] > 0, 0)

                # Rinomina colonne medie per chiarezza nella tabella
                segment_stats.rename(columns={
                    'mio_ribasso_percentuale': 'Tuo Rib. Medio %',
                    'ribasso_aggiudicatario_percentuale': 'Agg. Rib. Medio %',
                    'soglia_anomalia_calcolata': 'Soglia Media %',
                    'numero_concorrenti': 'Num. Conc. Medio'
                }, inplace=True)

                # Seleziona e ordina colonne per la visualizzazione finale
                display_cols = ['Num. Gare', 'Vinte', 'Partecipate (A/P)', 'Win Rate (%)',
                                'Tuo Rib. Medio %', 'Agg. Rib. Medio %', 'Soglia Media %', 'Num. Conc. Medio']
                # Mantieni solo le colonne effettivamente calcolate
                segment_stats_display = segment_stats[[col for col in display_cols if col in segment_stats.columns]]

                # Visualizza la tabella con stile e formattazione
                st.dataframe(segment_stats_display.style.format({
                    'Win Rate (%)': '{:.1f}%', # 1 decimale per win rate
                    'Tuo Rib. Medio %': PERCENTAGE_DISPLAY_FORMAT, # 4 decimali per medie %
                    'Agg. Rib. Medio %': PERCENTAGE_DISPLAY_FORMAT,
                    'Soglia Media %': PERCENTAGE_DISPLAY_FORMAT,
                    'Num. Conc. Medio': '{:.1f}' # 1 decimale per media concorrenti
                }).highlight_max(subset=['Win Rate (%)'], color='lightgreen', axis=0) # Evidenzia max win rate
                  .highlight_min(subset=['Tuo Rib. Medio %', 'Agg. Rib. Medio %'], color='lightblue', axis=0) # Evidenzia min ribassi medi
                  , use_container_width=True)

                # Grafico opzionale: Win Rate per Segmento
                if not segment_stats_display.empty and 'Win Rate (%)' in segment_stats_display.columns:
                     plot_df = segment_stats_display.reset_index() # Porta il segmento da indice a colonna per Plotly
                     # Ordina per Num. Gare per possibile visualizzazione migliore
                     plot_df = plot_df.sort_values(by='Num. Gare', ascending=False)
                     fig_segment = px.bar(plot_df, x=segment_col, y='Win Rate (%)',
                                          color='Num. Gare', # Colora barre per numero gare nel segmento
                                          color_continuous_scale=px.colors.sequential.Viridis, # Scala colori
                                          title=f"Win Rate per {segment_type}",
                                          labels={segment_col: segment_type, 'Win Rate (%)': 'Win Rate (%)'},
                                          hover_data=plot_df.columns # Mostra tutti i dati nel tooltip
                                          )
                     fig_segment.update_layout(yaxis_ticksuffix="%")
                     st.plotly_chart(fig_segment, use_container_width=True)

            except Exception as e_segment:
                 st.error(f"Errore durante l'analisi per segmenti: {e_segment}")
                 print(traceback.format_exc())
        st.divider()

    else:
        st.info("Nessuna gara corrisponde ai filtri applicati per visualizzare la dashboard analitica.")


    # --- Stima Soglia Statistica ---
    st.header("🔮 Stima Soglia Anomalia (Statistica)")
    st.markdown("Stima basata sui dati storici **filtrati**. Utile per avere un'idea del range probabile di soglia per gare simili a quelle visualizzate.")
    # Verifica se ci sono dati filtrati e la colonna soglia esiste e ha valori
    if not df_filtered.empty and 'soglia_anomalia_calcolata' in df_filtered.columns and df_filtered['soglia_anomalia_calcolata'].notna().any():
        # Calcola statistiche sulla colonna soglia (ignorando NaN)
        df_pred_base = df_filtered.dropna(subset=['soglia_anomalia_calcolata']).copy()
        if not df_pred_base.empty:
            st.write(f"Stima calcolata su **{len(df_pred_base)}** gare filtrate con soglia nota.")
            # Calcolo statistiche descrittive
            soglia_media = df_pred_base['soglia_anomalia_calcolata'].mean()
            soglia_mediana = df_pred_base['soglia_anomalia_calcolata'].median()
            soglia_std_dev = df_pred_base['soglia_anomalia_calcolata'].std()
            soglia_min = df_pred_base['soglia_anomalia_calcolata'].min()
            soglia_max = df_pred_base['soglia_anomalia_calcolata'].max()

            # Visualizza metrica principale (Media)
            st.metric("Soglia Media Stimata (sui dati filtrati)", PERCENTAGE_METRIC_FORMAT.format(soglia_media),
                      help="Media aritmetica delle soglie di anomalia calcolate per le gare filtrate.")

            # Visualizza altre statistiche in colonne
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            col_stats1.info(f"Mediana: {soglia_mediana:.4f}%")
            col_stats1.caption("Valore centrale (50° percentile).")
            # Mostra dev. std solo se calcolabile (più di 1 campione)
            if len(df_pred_base) > 1 and pd.notna(soglia_std_dev):
                col_stats2.info(f"Dev. Std: {soglia_std_dev:.4f}%")
                col_stats2.caption("Misura della dispersione dei valori.")
            else:
                col_stats2.info("Dev. Std: N/A")
            col_stats3.info(f"Range: {soglia_min:.4f}% - {soglia_max:.4f}%")
            col_stats3.caption("Valore minimo e massimo osservati.")

            # Istogramma distribuzione soglie
            st.markdown("**Distribuzione delle Soglie (sui dati filtrati)**")
            fig_hist_soglie = px.histogram(df_pred_base, x='soglia_anomalia_calcolata', nbins=15, # Adeguato numero di bin
                                           title="Distribuzione Soglie Anomalia Gare Filtrate")
            fig_hist_soglie.update_layout(
                xaxis_title="Soglia Anomalia Calcolata (%)",
                yaxis_title="Numero Gare",
                xaxis_tickformat=PERCENTAGE_FORMAT, # Formato asse X
                xaxis_ticksuffix="%" # Simbolo % asse X
            )
            st.plotly_chart(fig_hist_soglie, use_container_width=True)
            st.caption("Questo istogramma mostra la frequenza dei diversi valori di soglia presenti nei dati filtrati.")
        else:
            # Questo caso non dovrebbe accadere se il check iniziale passa, ma per sicurezza...
            st.warning("Nessuna gara con valore di soglia valido trovata nei dati filtrati.")
    else:
        # Messaggi specifici se non si può calcolare la stima
        if df_filtered.empty:
            st.warning("Nessuna gara selezionata dai filtri per calcolare la stima statistica della soglia.")
        elif 'soglia_anomalia_calcolata' not in df_filtered.columns:
            st.warning("Colonna 'soglia_anomalia_calcolata' non presente nei dati per calcolare la stima.")
        else: # Colonna esiste ma non ha valori validi
            st.warning("Nessun valore valido per 'soglia_anomalia_calcolata' trovato nei dati filtrati per calcolare la stima.")

# --- FINE BLOCCO ELSE (quando df_gare non è vuoto) ---
st.divider()

# --- Modulo Machine Learning ---
st.header("🤖 Modulo Previsione Avanzata (Machine Learning)")
st.markdown("Utilizza un modello predittivo (Random Forest) per stimare la soglia di anomalia basandosi sulle caratteristiche della gara. Richiede addestramento preliminare.",
            help="Il modello usa Importo Base, Categoria, Num. Concorrenti, Anno/Mese Gara.")

# Carica TUTTI i dati per training/info ML (non filtrati)
df_gare_ml = db_utils.get_all_gare(_refresh_trigger=st.session_state.refresh_data)

# Condizioni per poter addestrare il modello
target_col='soglia_anomalia_calcolata'
# Feature minime richieste per un addestramento sensato (oltre al target)
required_feature_cols = ['importo_base', 'data_gara'] # Categoria/Num.Conc. sono usate ma potrebbero mancare/essere imputate
min_samples_for_train = 15 # Numero minimo di campioni validi
can_train = False

# Verifica se ci sono abbastanza dati validi
if not df_gare_ml.empty and target_col in df_gare_ml.columns and all(col in df_gare_ml.columns for col in required_feature_cols):
    # Conta campioni dove SIA target CHE features richieste NON sono NaN
    valid_samples = df_gare_ml.dropna(subset=[target_col] + required_feature_cols).shape[0]
    if valid_samples >= min_samples_for_train:
        can_train = True
    else:
        st.warning(f"Dati storici insufficienti per addestrare il modello ML. Servono almeno {min_samples_for_train} gare con '{target_col}', '{required_feature_cols[0]}' e '{required_feature_cols[1]}' validi. Trovati: {valid_samples}.")
else:
    st.warning(f"Dati storici insufficienti o colonne essenziali mancanti ('{target_col}', '{required_feature_cols[0]}', '{required_feature_cols[1]}') per addestrare il modello ML.")

# --- Sezione Addestramento (se possibile) ---
if can_train:
    with st.expander("🔧 Addestra / Riaddestra Modello di Previsione Soglia"):
        st.markdown("Addestra un modello *Random Forest* usando tutti i dati storici disponibili nel database che hanno una soglia di anomalia nota e le feature richieste. Il modello impara a predire la `soglia_anomalia_calcolata` basandosi su `importo_base`, `categoria_lavori`, `numero_concorrenti`, anno e mese della gara.")
        st.caption("L'addestramento sovrascrive eventuali modelli precedenti e può richiedere qualche istante.")

        if st.button("🚀 Avvia Addestramento Modello", key="train_ml_button"):
            with st.spinner("Addestramento modello ML in corso..."):
                # Passa l'intero DataFrame (la funzione train_model farà il preprocessing)
                results = ml_utils.train_model(df_gare_ml)

            if results:
                st.success("Addestramento modello completato e salvato!")
                # Mostra metriche di valutazione sul test set
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.metric("Errore Medio Assoluto (MAE)", f"{results['mae']:.4f}%", delta=None,
                              help="Errore medio di previsione sul test set (in punti percentuali). Più basso è, meglio è.")
                with col_met2:
                    st.metric("Coefficiente R²", f"{results['r2']:.3f}", delta=None,
                              help="Indica quanto bene il modello spiega la varianza della soglia (0-1). Più vicino a 1 è, meglio è.")
                st.caption(f"Modello addestrato usando {results['n_samples']} campioni validi (dopo preprocessing).")

                # Mostra importanza delle feature se disponibile
                if 'feature_importances' in results and not results['feature_importances'].empty:
                    st.divider()
                    st.subheader("Importanza delle Feature nel Modello")
                    st.markdown("Quanto ogni fattore ha contribuito alla previsione della soglia nel modello appena addestrato (valori più alti indicano maggiore importanza).")
                    try:
                        df_imp = results['feature_importances'].reset_index()
                        df_imp.columns = ['Feature', 'Importanza']
                        # Pulisci nomi feature per leggibilità (es: 'anno_gara' -> 'Anno Gara')
                        df_imp['Feature'] = df_imp['Feature'].str.replace('_', ' ').str.title()
                        # Grafico a barre orizzontale
                        fig_imp = px.bar(df_imp.sort_values(by='Importanza', ascending=True), # Ordina per vedere meglio
                                         x='Importanza', y='Feature', orientation='h',
                                         title="Importanza Feature per Previsione Soglia",
                                         labels={'Importanza': 'Importanza Relativa (Gini Importance)'},
                                         height=max(400, len(df_imp) * 30)) # Altezza dinamica
                        fig_imp.update_layout(xaxis_tickformat=".1%") # Formatta asse x come percentuale
                        st.plotly_chart(fig_imp, use_container_width=True)
                    except Exception as e_imp:
                        st.warning(f"Impossibile visualizzare l'importanza delle feature: {e_imp}")

                # Potrebbe essere utile ricaricare per assicurarsi che il form di predizione veda il nuovo modello
                st.rerun()
            else:
                st.error("Addestramento del modello ML fallito. Controllare i log o i dati nel database.")

# --- Sezione Previsione (se il modello esiste) ---
model_exists = os.path.exists(ml_utils.MODEL_PATH)
if not model_exists:
    st.info("Il modello di previsione ML non è stato ancora addestrato. Addestralo usando l'opzione sopra (se i dati sono sufficienti).")
else:
    st.subheader("🔮 Prevedi Soglia per Nuova Gara (con ML)")
    st.markdown("Inserisci i dati stimati di una nuova gara per ottenere una previsione della soglia di anomalia basata sul modello Machine Learning addestrato.")

    with st.form("ml_prediction_input_form"):
        c1_pred, c2_pred = st.columns(2)
        with c1_pred: # Input principali
            pred_importo = st.number_input("Importo Base (€) Stimato*", value=None, format=CURRENCY_INTERNAL_FORMAT, step=1000.0, key="pred_imp", placeholder="Es: 250000.00", help="Importo base stimato della nuova gara.")
            pred_data = st.date_input("Data Gara Stimata*", value=datetime.date.today(), key="pred_data", help="Data di riferimento stimata (influenza anno/mese usati dal modello).")
        with c2_pred: # Input secondari/stimati
            # Carica categorie valide DAL MODELLO salvato per coerenza
            _, _, saved_encoders = ml_utils.load_model_and_dependencies()
            categorie_valide_modello = ["Sconosciuto"] # Opzione di default
            if saved_encoders and 'categoria_lavori' in saved_encoders:
                 # Prendi le classi dall'encoder salvato
                 categorie_valide_modello.extend(list(saved_encoders['categoria_lavori'].classes_))
                 # Rimuovi eventuale 'Sconosciuto' duplicato se già presente nelle classi
                 categorie_valide_modello = sorted(list(set(categorie_valide_modello)))
                 if 'Sconosciuto' not in categorie_valide_modello: categorie_valide_modello.insert(0, 'Sconosciuto')
            else:
                 # Fallback: usa categorie dai dati grezzi (meno ideale ma funziona)
                 categorie_valide_modello.extend(sorted(df_gare_ml['categoria_lavori'].dropna().astype(str).unique().tolist()))
                 categorie_valide_modello = sorted(list(set(categorie_valide_modello)))
                 if 'Sconosciuto' not in categorie_valide_modello: categorie_valide_modello.insert(0, 'Sconosciuto')

            pred_categoria = st.selectbox("Categoria Stimata", options=categorie_valide_modello, index=0, key="pred_cat", help="Categoria stimata (seleziona 'Sconosciuto' se non nota). Deve essere una categoria vista durante l'addestramento.")
            # Stima numero concorrenti (usa mediana storica come default sensato)
            num_conc_med = df_gare_ml['numero_concorrenti'].median() if 'numero_concorrenti' in df_gare_ml and df_gare_ml['numero_concorrenti'].notna().any() else 10
            pred_num_conc = st.number_input("Num. Concorrenti Stimato", value=int(num_conc_med) if pd.notna(num_conc_med) else 10, min_value=1, step=1, key="pred_conc", help="Numero stimato di concorrenti (influenza la previsione).")

        # Bottone per avviare la previsione
        predict_button = st.form_submit_button("⚡ Prevedi Soglia con ML", use_container_width=True, type="primary")

        if predict_button:
            # Validazione input per previsione
            if pred_importo is None or pred_data is None:
                st.error("Importo Base e Data Gara sono obbligatori per la previsione.")
            else:
                # Prepara il dizionario di feature per la funzione di predizione
                input_features = {
                   'importo_base': float(pred_importo),
                   'data_gara': pred_data.strftime(DATE_FORMAT_STR), # Passa data come stringa YYYY-MM-DD
                   'categoria_lavori': pred_categoria, # Passa la categoria selezionata
                   'numero_concorrenti': int(pred_num_conc)
                   }
                # Chiama la funzione di previsione
                prediction = ml_utils.predict_soglia(input_features)
                # Mostra il risultato se la previsione ha successo
                if prediction is not None:
                    st.success(f"**Previsione Soglia ML Stimata: {prediction:.4f}%**")
                    st.caption("Nota: Questa è una stima basata sul modello ML attualmente addestrato e sui dati forniti.")
                # Gli errori vengono gestiti e mostrati da predict_soglia tramite st.error

# --- Footer ---
st.sidebar.divider()
st.sidebar.caption(f"Analisi Gare Appalto v1.7 - DB: {db_utils.DB_FILENAME}")