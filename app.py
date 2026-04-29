from fastapi import FastAPI, UploadFile
from rag import add_document, search

import ollama  

app = FastAPI()


@app.post("/upload")
async def upload(file: UploadFile):
    content = await file.read()
    text = content.decode()

    add_document(file.filename, text)

    return {"message": "uploaded + indexed"}


@app.get("/ask")
def ask(question: str):

  
    context_chunks = search(question, k=3)
    context = "\n".join(context_chunks)

  
    prompt = f"""
Answer by using the rules set below.

Context:
{context}

Question:
{question}

If the answer is not within the rules say "I'm sorry, I don't understand".
"""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        answer_text = response["message"]["content"]

    except:

        answer_text = context

    return {
        "answer": answer_text,
        "sources": context_chunks
    }
