# Importar el módulo necesario
from agents.articles_agent import ArticlesAgent

# Crear una instancia de la clase ArticlesAgent
articles_agent = ArticlesAgent()

# Definir el tema de entrada
input_topic = "The king Charles crowning"

# Llamar al método run_chain en la instancia de ArticlesAgent
result = articles_agent.run_agent()

# Imprimir el artículo generado
print(result)