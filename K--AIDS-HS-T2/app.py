from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences



import pickle
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Global variables
model = None
tokenizer = None

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    
    model_path = "best_model.h5"
    tokenizer_path = "tokenizer.pickle"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    yield

# Initialize FastAPI app
app = FastAPI(
    title="Twitter Sentiment Analysis",
    description="API and frontend for predicting sentiment of tweets",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for serving index.html and twitter_bird.png
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at the root
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# Define request body model
class TweetRequest(BaseModel):
    text: str

# Define prediction function
def predict_sentiment(text: str):
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len = 50
    
    xt = tokenizer.texts_to_sequences([text])
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    yt = model.predict(xt).argmax(axis=1)
    
    return sentiment_classes[yt[0]]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict")
async def predict(tweet: TweetRequest):
    try:
        prediction = predict_sentiment(tweet.text)
        return {
            "text": tweet.text,
            "sentiment": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")  