from langchain_community.llms.ollama import Ollama


llm = Ollama(model="phi")
output = llm.invoke("Erzähle mir einen Witz, halte dich kurz!")

print (output)