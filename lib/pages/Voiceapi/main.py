from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import speech_recognition as sr
from langchain import PromptTemplate, LLMChain
from langchain_google_genai import GoogleGenerativeAI
import uuid
import os

app = FastAPI()

GEMINI_API_KEY = "Your API KEY"
llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)

prompt_template = PromptTemplate(
    input_variables=["query"],
    template="You are a compassionate and knowledgeable mental health assistant, "
        "specialized in helping brain stroke rehabilitation patients. "
        "Provide clear, supportive, and motivating responses that guide users "
        "in their recovery journey. Keep the tone positive and reassuring.\n\n"
        "User Query: {query}\n"
        "Assistant Response:"
)

chain = LLMChain(llm=llm, prompt=prompt_template)

class UserQuery(BaseModel):
    query: str

@app.post("/chat")
async def chat(user_query: UserQuery):
    try:
        response = chain.run(user_query.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    unique_filename = f"temp_audio_{uuid.uuid4()}.wav"

    try:
        with open(unique_filename, "wb") as f:
            f.write(await file.read())

        with sr.AudioFile(unique_filename) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            os.remove(unique_filename)
            return {"error": "Could not understand the audio"}
        except sr.RequestError:
            os.remove(unique_filename)
            return {"error": "Could not request results from Google Speech API"}

        os.remove(unique_filename)  

        response = chain.run(text)
        return {"text_input": text, "response": response}

    except Exception as e:
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
        raise HTTPException(status_code=500, detail=str(e))
