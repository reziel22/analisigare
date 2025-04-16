import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import numpy as np
import streamlit as st
import traceback
import datetime

# --- Costanti ---
MODEL_DIR = "ml_model"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_soglia_model.joblib")
COLUMNS_PATH = os.path.join(MODEL_DIR, "model_columns.joblib")
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.joblib")
os.makedirs(MODEL_DIR, exist_ok=True) # Crea la directory se non esiste

# --- Funzioni ---
def preprocess_data(df, fit_encoders=False, saved_encoders=None, saved_columns=None):
    """
    Preprocessa i dati per il modello ML.
    Gestisce: selezione feature, conversione date, imputazione NaN, encoding categorico.

    Args:
        df (pd.DataFrame): DataFrame di input.
        fit_encoders (bool): Se True, adatta nuovi LabelEncoder e li salva.
                             Se False, usa encoders salvati per trasformare.
        saved_encoders (dict): Dizionario di LabelEncoder salvati (usato se fit_encoders=False).
        saved_columns (list): Lista di colonne attese dal modello (usato se fit_encoders=False).

    Returns:
        tuple: (X, y, encoders, columns)
               X: DataFrame delle feature processate.
               y: Series del target (o None se non presente/fit_encoders=False).
               encoders: Dizionario degli encoder (solo se fit_encoders=True).
               columns: Lista delle colonne di X (solo se fit_encoders=True).
               Restituisce (None, None, None, None) in caso di errore grave.
    """
    print(f"Inizio preprocessing ML (fit_encoders={fit_encoders})...")
    target = 'soglia_anomalia_calcolata'
    # Feature iniziali considerate dal modello
    initial_features = ['importo_base', 'categoria_lavori', 'numero_concorrenti', 'data_gara']

    try:
        # Seleziona solo le colonne rilevanti presenti nel DataFrame
        cols_to_keep = [f for f in initial_features if f in df.columns]
        if target in df.columns:
            cols_to_keep.append(target)
        df_proc = df[list(set(cols_to_keep))].copy() # Usa set per evitare duplicati
        print(f"Colonne iniziali selezionate: {df_proc.columns.tolist()}")

        # Gestione Target (solo se presente e in modalità training)
        if target in df_proc.columns and fit_encoders:
            initial_rows = len(df_proc)
            df_proc.dropna(subset=[target], inplace=True)
            print(f"Rimosse {initial_rows - len(df_proc)} righe per target NaN.")
            if df_proc.empty:
                print("Nessun dato valido rimasto dopo rimozione target NaN."); return None, None, None, None
            y = df_proc[target]
        else:
            y = None # Non c'è target o non siamo in training

        # Feature Engineering: Data -> Anno/Mese
        if 'data_gara' in df_proc.columns:
            df_proc['data_gara'] = pd.to_datetime(df_proc['data_gara'], errors='coerce')
            # Crea features solo se ci sono date valide
            if df_proc['data_gara'].notna().any():
                df_proc['anno_gara'] = df_proc['data_gara'].dt.year
                df_proc['mese_gara'] = df_proc['data_gara'].dt.month
                print("Create features 'anno_gara' e 'mese_gara'.")
            # Rimuovi colonna data originale dopo aver estratto le features
            df_proc.drop('data_gara', axis=1, inplace=True, errors='ignore')

        # Imputazione NaN per Feature Numeriche
        numeric_features = list(df_proc.select_dtypes(include=np.number).columns)
        if target in numeric_features: # Non imputare il target
            numeric_features.remove(target)
        for col in numeric_features:
            if df_proc[col].isnull().any():
                # Usa mediana per robustezza agli outlier, fallback a 0 se mediana è NaN
                median_val = df_proc[col].median()
                fill_value = median_val if pd.notna(median_val) else 0
                df_proc[col].fillna(fill_value, inplace=True)
                print(f"Imputati NaN in '{col}' numerica con {fill_value}.")

        # Encoding Feature Categoriche e Imputazione NaN
        categorical_features = list(df_proc.select_dtypes(include=['object', 'category']).columns)
        encoders = {}
        if saved_encoders: # Carica encoder se forniti (modalità predizione)
            encoders = saved_encoders

        for col in categorical_features:
            # Imputa NaN con una categoria specifica 'Sconosciuto'
            if df_proc[col].isnull().any():
                df_proc[col].fillna('Sconosciuto', inplace=True)
                print(f"Imputati NaN in '{col}' categorica con 'Sconosciuto'.")
            # Assicura tipo stringa per LabelEncoder
            df_proc[col] = df_proc[col].astype(str)

            if fit_encoders: # Modalità Training: Adatta e salva encoder
                le = LabelEncoder()
                df_proc[col] = le.fit_transform(df_proc[col])
                encoders[col] = le # Salva l'encoder addestrato
                print(f"Label Encoding (fit) applicato a '{col}'. Classi: {le.classes_[:5]}...") # Mostra alcune classi
            else: # Modalità Predizione: Usa encoder salvato
                if col in encoders:
                    le = encoders[col]
                    # Trasforma usando l'encoder salvato. Gestisci classi non viste in training.
                    df_proc[col] = df_proc[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1) # Assegna -1 a classi sconosciute
                    print(f"Label Encoding (transform) applicato a '{col}'.")
                else:
                    # Se manca un encoder per una colonna attesa, non possiamo usarla
                    print(f"Attenzione: Encoder per '{col}' non trovato durante la trasformazione. Colonna rimossa.")
                    df_proc.drop(col, axis=1, inplace=True, errors='ignore')

        # Seleziona le feature finali (tutte le colonne tranne il target)
        final_features_processed = [col for col in df_proc.columns if col != target]
        X = df_proc[final_features_processed]
        print(f"Features finali nel DataFrame processato: {X.columns.tolist()}")

        # Allineamento Colonne (solo in modalità Predizione)
        if not fit_encoders and saved_columns:
            print(f"Colonne attese dal modello salvato: {saved_columns}")
            missing_cols = set(saved_columns) - set(X.columns)
            for c in missing_cols:
                print(f"Aggiunta colonna mancante '{c}' con valore 0.")
                X[c] = 0 # Aggiungi colonne mancanti con 0 (o altra logica se necessario)
            # Riordina e seleziona colonne per matchare quelle del training
            extra_cols = set(X.columns) - set(saved_columns)
            if extra_cols:
                print(f"Rimozione colonne extra non presenti nel training: {list(extra_cols)}")
                X.drop(columns=list(extra_cols), inplace=True, errors='ignore')

            try:
                 X = X[saved_columns] # Assicura ordine corretto
                 print(f"Colonne allineate all'ordine del modello: {X.columns.tolist()}")
            except KeyError as e:
                 print(f"Errore critico durante l'allineamento delle colonne: {e}. Colonne disponibili: {X.columns.tolist()}")
                 st.error(f"Errore nell'allineamento delle colonne del modello: {e}. Verifica i dati di input.")
                 return None, None, None, None

        print("Preprocessing ML completato con successo.")
        # Restituisci X, y (se applicabile), encoders (se fit), colonne finali (se fit)
        return X, y, (encoders if fit_encoders else None), (X.columns.tolist() if fit_encoders else None)

    except Exception as e:
        print(f"Errore grave durante preprocess_data: {e}")
        st.error(f"Errore durante il preprocessing dei dati ML: {e}")
        traceback.print_exc();
        return None, None, None, None

def train_model(df):
    """
    Addestra un modello RandomForestRegressor sui dati forniti.
    Salva il modello, le colonne e gli encoder.

    Args:
        df (pd.DataFrame): DataFrame contenente i dati storici.

    Returns:
        dict: Dizionario con metriche ('mae', 'r2'), numero campioni ('n_samples'),
              e importanza feature ('feature_importances') se l'addestramento ha successo.
              None altrimenti.
    """
    if df.empty:
        st.error("Impossibile addestrare: DataFrame di input vuoto."); return None

    print("Avvio processo di addestramento modello...")
    with st.spinner("Preprocessing dati per addestramento..."):
        X, y, encoders, trained_columns = preprocess_data(df, fit_encoders=True)

    if X is None or y is None or X.empty or y.empty:
        st.error("Addestramento fallito: Dati insufficienti o errore durante il preprocessing."); return None
    if not trained_columns:
        st.error("Addestramento fallito: Nessuna feature valida identificata dopo il preprocessing."); return None
    if not encoders:
         print("Attenzione: Nessun encoder categorico addestrato (potrebbe essere normale se non ci sono feature categoriche).")

    print(f"Dati pronti per l'addestramento. Numero campioni: {len(X)}, Numero feature: {len(trained_columns)}")
    with st.spinner("Addestramento modello RandomForest..."):
        try:
            # Suddivisione dati in set di training e test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Dimensioni set - Training: {X_train.shape}, Test: {X_test.shape}")

            # Inizializzazione e addestramento modello
            # Parametri possono essere ottimizzati (es. con GridSearchCV)
            rf_model = RandomForestRegressor(n_estimators=100, # Numero alberi
                                             random_state=42,
                                             n_jobs=-1, # Usa tutti i core CPU
                                             max_depth=10, # Limita profondità alberi per evitare overfitting
                                             min_samples_split=5, # Min campioni per splittare nodo
                                             min_samples_leaf=3) # Min campioni in nodo foglia

            rf_model.fit(X_train, y_train)

            # Valutazione sul Test Set
            y_pred = rf_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Valutazione Modello su Test Set - MAE: {mae:.4f}%, R2: {r2:.4f}")

            # Salvataggio modello, colonne e encoder
            joblib.dump(rf_model, MODEL_PATH)
            joblib.dump(trained_columns, COLUMNS_PATH)
            joblib.dump(encoders, LABEL_ENCODERS_PATH)
            print(f"Modello, colonne e encoder salvati con successo nella directory: {MODEL_DIR}")

            # *** Estrai e restituisci feature importances ***
            feature_importances = pd.Series(rf_model.feature_importances_, index=trained_columns).sort_values(ascending=False)
            print("Importanza delle feature calcolata.")

            return {
                "mae": mae,
                "r2": r2,
                "n_samples": len(X), # Numero campioni usati DOPO preprocessing
                "feature_importances": feature_importances # Serie pandas con importanza feature
                }
        except Exception as e:
            st.error(f"Errore critico durante l'addestramento o salvataggio del modello: {e}")
            print(traceback.format_exc());
            return None

def load_model_and_dependencies():
    """Carica il modello, la lista delle colonne e gli encoder salvati."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, COLUMNS_PATH, LABEL_ENCODERS_PATH]):
        print("File del modello o delle dipendenze non trovati.")
        # Non mostrare errore qui, verrà gestito nel chiamante (es. predict_soglia)
        return None, None, None
    try:
        model = joblib.load(MODEL_PATH)
        columns = joblib.load(COLUMNS_PATH)
        encoders = joblib.load(LABEL_ENCODERS_PATH)
        print("Modello, colonne e encoder caricati da disco.")
        return model, columns, encoders
    except Exception as e:
        print(f"Errore durante il caricamento del modello o delle dipendenze: {e}")
        st.error(f"Errore nel caricamento del modello ML salvato: {e}")
        traceback.print_exc();
        return None, None, None

def predict_soglia(input_data_dict):
    """
    Esegue una previsione della soglia usando il modello salvato.

    Args:
        input_data_dict (dict): Dizionario contenente i valori delle feature
                                per la gara di cui prevedere la soglia.
                                Es: {'importo_base': ..., 'data_gara': 'YYYY-MM-DD', ...}

    Returns:
        float: Valore previsto della soglia (in percentuale).
               None se la previsione fallisce o il modello non è pronto.
    """
    print(f"Avvio previsione soglia ML per input: {input_data_dict}")
    # Carica modello e dipendenze necessarie
    model, saved_columns, saved_encoders = load_model_and_dependencies()

    if model is None or saved_columns is None or saved_encoders is None:
        st.error("Impossibile eseguire la previsione: Modello ML non caricato o incompleto.")
        return None

    # Converti il dizionario di input in DataFrame (con una sola riga)
    input_df = pd.DataFrame([input_data_dict])

    # Preprocessa i dati di input usando gli encoder e le colonne salvate
    # Utilizza fit_encoders=False per applicare le trasformazioni salvate
    with st.spinner("Preprocessing dati per previsione..."):
        X_pred, _, _, _ = preprocess_data(input_df,
                                          fit_encoders=False,
                                          saved_encoders=saved_encoders,
                                          saved_columns=saved_columns)

    if X_pred is None or X_pred.empty:
        st.error("Previsione fallita: Errore durante il preprocessing dei dati di input.")
        return None

    # Esegui la previsione
    try:
        with st.spinner("Esecuzione previsione ML..."):
            prediction = model.predict(X_pred)
            predicted_value = prediction[0] # predict ritorna un array
            print(f"Previsione ML eseguita con successo. Risultato: {predicted_value:.4f}%")
            return predicted_value
    except Exception as e:
        st.error(f"Errore durante l'esecuzione della previsione ML: {e}")
        print(f"Errore durante model.predict(): {e}")
        print("Dati passati al modello (prime 5 righe se multiple):")
        try:
            print(X_pred.head().to_markdown())
            print("Tipi di dati:")
            print(X_pred.dtypes)
        except Exception as dump_err:
             print(f"Impossibile stampare i dati di input: {dump_err}")
        traceback.print_exc();
        return None