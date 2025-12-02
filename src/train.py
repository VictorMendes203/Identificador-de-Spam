import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    """
    Função principal para carregar os dados, treinar o modelo
    e salvar os artefatos (modelo e vetorizador).
    """
    
    print("--- [Iniciando Script de Treinamento] ---")

    #Definir Caminhos
    # __file__ é o caminho deste script (src/train.py)
  
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR) # Raiz do projeto 'Detector de Spam'

    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'emails.csv')
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'spam_model.pkl')
    VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

    print(f"Carregando dados de: {DATA_PATH}")

    #Carregar os Dados
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding='latin-1')
    
    print("Dados carregados com sucesso.")

    # --- 3. Preparar os Dados (X e y) ---
    # Usando o nome correto da coluna que descobrimos: 'spam'
    X = df['text']
    y = df['spam'] 

    #Aplicar "Bag of Words" (Vetorização)
    print("Aplicando Bag of Words (CountVectorizer)...")

    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(X)
    
    print(f"Vocabulário criado com {X_bow.shape[1]} palavras únicas.")

    #Dividir em Treino e Teste
  
    X_train, X_test, y_train, y_test = train_test_split(
        X_bow, y, test_size=0.20, random_state=42
    )

    #Treinar o Classificador (Naive Bayes)
    print("Treinando o modelo Naive Bayes...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Modelo treinado.")

    #Avaliar o Modelo
    print("Avaliando modelo no set de teste...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"--- Acurácia no Teste: {accuracy * 100:.2f}% ---")
    print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))
    print("="*40 + "\n")

    #Salvar os Artefatos
    
    os.makedirs(MODEL_DIR, exist_ok=True) 
    
    print(f"Salvando modelo em: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    
    print(f"Salvando vetorizador em: {VECTORIZER_PATH}")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print("--- [Treinamento Concluído com Sucesso] ---")


# --- Padrão Python para executar o script ---
if __name__ == "__main__":
    train_model()