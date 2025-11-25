from groq import Groq
from fastapi import FastAPI,UploadFile,File
from fastapi.responses import FileResponse
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
from pydantic import BaseModel
import os

app = FastAPI()

@app.get("/")
def home():
    return FileResponse("home.html")

@app.post("/upload")
async def extraction(JD: UploadFile = File(...), resume: UploadFile = File(...)):
    
    global resume_text, JD_text

    resume_path = f"./savepdf/{resume.filename}"
    with open(resume_path, "wb") as f:
        f.write(await resume.read())
        resume_text = extract_text(resume_path)
        
    JD_path = f"./savepdf/{JD.filename}"
    with open(JD_path, "wb") as f:
        f.write(await JD.read())
        JD_text = extract_text(JD_path)
    return {"resume":resume_text ,"JD":JD_text}

class InputData(BaseModel):
    resume_text: str
    JD_text: str


@app.post("/model")

def model(data:InputData):
    resume_text = data.resume_text
    JD_text = data.JD_text
    load_dotenv()
    APIKEY = os.getenv("APIKEY")
    system_prompt = f"""
    based on the job description: {JD_text} review the resume:{resume_text}
    
    You are an expert ATS and Resume Optimization Assistant.

Your task is to:
1. Read the Job Description (JD) and extract:
   - Key responsibilities
   - Required skills
   - Preferred skills
   - Domain-specific keywords
   - Tools/technologies mentioned
2. Read the candidate’s resume and extract:
   - Skills
   - Experience points
   - Achievements
   - Tools & technologies
3. Compare JD and resume using semantic understanding.
4. Output clear, structured suggestions including:
   - Missing skills or keywords the resume should include
   - Experience points that should be rephrased to better match the JD
   - Irrelevant content that can be removed
   - New bullet points the candidate can add
   - ATS optimization tips (action verbs, measurable results, keyword placement)
5. Ensure suggestions are specific, actionable, and tailored to the given JD.
6. NEVER modify the resume unless explicitly asked—only provide suggestions.
7. Final output format:
   - JD Summary
   - Resume Summary
   - Skill Match Analysis
   - Missing Skills / Gaps
   - Suggested Resume Improvements
   - Suggested New Bullet Points
   - ATS Score Estimate (0–100)"""

    client = Groq(api_key=APIKEY)
    completion = client.chat.completions.create(
      model="openai/gpt-oss-20b",
      messages=[
      {
        "role": "user",
        "content": system_prompt
      }
    ],
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None
  )

    fullanswer = ""
    for chunk in completion:fullanswer += chunk.choices[0].delta.content or ""
    return fullanswer

