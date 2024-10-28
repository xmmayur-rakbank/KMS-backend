import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, List
from pydantic import BaseModel
from training import training
from chat import chatting
from delete import deleting
from list import listing

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    try:
        # logger.log(msg="Hello!, Welcome Message",level=logging.INFO)
        return {"Hello": "Hello from API"}
    except Exception as e:
        # logger.log(msg= f"An error occurred: {str(e)}",level=logging.ERROR)
        return {"message": str(e)}
    
@app.post("/train/")
async def train(departmant: Annotated[str, Form()], pdf_files: List[UploadFile]):
    try:
        for file in pdf_files:
            await training(file, departmant)
        # logger.log(msg="List of all output files extracted for usecase 1",level=logging.INFO)
        return "training successful"
    except Exception as e:
        # logger.log(msg= f"An error occurred: {str(e)}",level=logging.ERROR)
        print(f"Exception in training() : {e}")
        raise HTTPException(status_code=500, detail=f"Something Went Wrong! Please try again!! {str(e)}")
    
class ChatMessage(BaseModel):
    user: str
    assistant: str

class ChatRequest(BaseModel):
    Query: str
    ChatHistory: List[ChatMessage] 

# async def chat(username:Annotated[str, Form()], departmant: list[Annotated[str, Form()]], is_followup: Annotated[bool, Form()], chatRequest: ChatRequest = Body(...)):
@app.post("/chat/")
async def chat(username:str, departmant: list[str] , chatRequest: ChatRequest, is_followup: bool):
    try:
        print(chatRequest)
        answer = await chatting(chatRequest.Query, departmant, chatRequest, is_followup, username)
        # chat_history_for_search = get
        return answer
    except Exception as e:
        # logger.log(msg= f"An error occurred: {str(e)}",level=logging.ERROR)
        print(f"Exception in chat() : {e}")
        raise HTTPException(status_code=500, detail=f"Something Went Wrong! Please try again!! {str(e)}")

@app.post("/delete/")
async def delete(filename: Annotated[str, Form()], username: Annotated[str, Form()]):
    try:
        await deleting(filename, username)
        # logger.log(msg="List of all output files extracted for usecase 1",level=logging.INFO)
        return "deleted successfully"
    except Exception as e:
        # logger.log(msg= f"An error occurred: {str(e)}",level=logging.ERROR)
        print(f"Exception in training() : {e}")
        raise HTTPException(status_code=500, detail=f"Something Went Wrong! Please try again!! {str(e)}")
 
@app.get("/list/")
async def list():
    try:
        results = await listing()
        # logger.log(msg="List of all output files extracted for usecase 1",level=logging.INFO)
        return results
    except Exception as e:
        # logger.log(msg= f"An error occurred: {str(e)}",level=logging.ERROR)
        print(f"Exception in training() : {e}")
        raise HTTPException(status_code=500, detail=f"Something Went Wrong! Please try again!! {str(e)}")
 
from fastapi.responses import RedirectResponse

@app.get("/load-frontend")
async def load_frontend():
    # Redirect to the frontend running on port 3000
    return RedirectResponse(url="http://localhost:3000")

if __name__ == "__main__":
    uvicorn.run(app)
