import joblib
import os
import sys

def predict_email(email_text):
    """
    Carrega o vetorizador e o modelo salvos para classificar
    um novo texto de e-mail.
    """
    
    #Definir Caminhos
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR) # Raiz do projeto 'Detector de Spam'

    MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'spam_model.pkl')
    VECTORIZER_PATH = os.path.join(ROOT_DIR, 'models', 'vectorizer.pkl')

    #Verificar se os arquivos de modelo existem
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("="*50)
        print("ERRO: Arquivos de modelo não encontrados!")
        print(f"Verifique se 'spam_model.pkl' e 'vectorizer.pkl' existem em:")
        print(f"{os.path.join(ROOT_DIR, 'models')}")
        print("\nRode o script 'train.py' primeiro:")
        print("python src/train.py")
        print("="*50)
        return

    #Carregar os Artefatos
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as e:
        print(f"Erro ao carregar os arquivos de modelo: {e}")
        return

    #Processar o novo e-mail
    email_bow = vectorizer.transform([email_text])
    
    #Fazer a Previsão
    prediction = model.predict(email_bow)
    
    #Mostrar o Resultado
    print("\n--- [Resultado da Classificação] ---")
    if prediction[0] == 1:
        print(">> Resultado: É SPAM! 🔴")
    else:
        print(">> Resultado: Não é Spam! ✅")
    print("="*37 + "\n")


#Padrão Python para executar o script
if __name__ == "__main__":
    # sys.argv é a lista de argumentos do terminal.
    # sys.argv[0] é o nome do script (predict.py)
    # sys.argv[1] é o primeiro argumento (o texto do e-mail)
    
    if len(sys.argv) < 2:
        print("\nERRO: Nenhum texto de e-mail fornecido.")
        print("Modo de usar:")
        print('python src/predict.py "Seu texto de e-mail aqui entre aspas"')
        print("\nExemplo:")
        print('python src/predict.py "congratulations you won a prize"')
    else:
        # Juntar todos os argumentos caso o texto não esteja entre aspas
        email_text_from_cli = " ".join(sys.argv[1:])
        predict_email(email_text_from_cli)