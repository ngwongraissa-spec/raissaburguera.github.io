import pandas as pd
import joblib


# --- CONFIGURATION PRODUCTION ---

scaler = joblib.load('scaler_billets.joblib')
log_reg = joblib.load('modele_logreg.joblib')
variables = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']


def detecteur_faux_billets(billets_production):
    try:
        # 1. Chargement des nouveaux billets envoyés par l'ONCFM
        df_prod = pd.read_csv('billets_production.csv')
        resultat = df_prod.copy()
        
        # 2. Préparation
        X_prod = df_prod[variables]
        
        # 3. Standardisation
        X_prod_scaled = scaler.transform(X_prod)
        
        # 4. Prédiction
        predictions = log_reg.predict(X_prod_scaled)
        probabilites = log_reg.predict_proba(X_prod_scaled)[:, 1] 
        
       # 5. Formatage
        resultat['Prediction'] = predictions
        resultat['Probabilité_Vrai'] = probabilites.round(4)

        # On définit le verdict selon le seuil statistique de 0.5
        # Si proba >= 0.5 alors 'VRAI BILLET', sinon 'FAUX BILLET'
        resultat['Verdict'] = ['VRAI BILLET' if p >= 0.5 else 'FAUX BILLET' for p in probabilites]
        
        return resultat[['id', 'Verdict', 'Probabilité_Vrai']]

    except Exception as e:
        return f"Erreur lors du traitement : {e}"

# --- EXÉCUTION ---
if __name__ == "__main__":
   
    nom_fichier_a_tester = "billets_production.csv" 
    
    resultats_finaux = detecteur_faux_billets(nom_fichier_a_tester)

    if isinstance(resultats_finaux, pd.DataFrame):
        print("✅ Analyse terminée avec succès.")
        print(resultats_finaux.head())
        stats = resultats_finaux['Verdict'].value_counts()
        # Exportation
        resultats_finaux.to_csv("resultats_expertise_ONCFM.csv", index=False)
        print("\n💾 Le fichier 'resultats_expertise_ONCFM.csv' a été généré.")
        print("\nRésumé de l'analyse :")
        print(f"- Nombre de billets authentiques : {stats.get('VRAI BILLET', 0)}")
        print(f"- Nombre de billets contrefaits : {stats.get('FAUX BILLET', 0)}")
    else:
        print(resultats_finaux)