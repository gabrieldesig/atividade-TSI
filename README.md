import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OllamaEmbeddings

# Define os documentos
documents = [ "Variáveis em Python e os tipos de dados básicos como int, float e string.", 
             "Estruturas de controle em Python: if, else, e elif para tomada de decisões.", 
             "Loops em Python, como for e while, para repetição de tarefas.", 
             "Funções em Python: como definir e chamar funções usando a palavra-chave def."]

# Define o modelo de embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
document_embeddings = embeddings.embed_documents(documents)
print(document_embeddings)
