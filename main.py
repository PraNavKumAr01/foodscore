from fastapi import FastAPI, UploadFile, File
import os
import json
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import boto3
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set AWS credentials and region
os.environ['AWS_ACCESS_KEY_ID'] = "AKIAUY4SWGEBYINYJTWI"
os.environ['AWS_SECRET_ACCESS_KEY'] = "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"
os.environ['AWS_DEFAULT_REGION'] = "ap-south-1"

app = FastAPI()

origins = [
    "http://localhost:3000",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow access from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Textract client
textract_client = boto3.client('textract')

os.environ["GOOGLE_API_KEY"] = "AIzaSyC9L2qLCTWBeQ9J2odAMz-yjXq3DzP6oOg"
llm = ChatGoogleGenerativeAI(model="gemini-pro")

def extract_text_from_image(image_bytes):
    # Call Textract to analyze the image and extract text
    response = textract_client.detect_document_text(Document={'Bytes': image_bytes})

    # Process the response to get extracted text
    extracted_text = []
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            extracted_text.append(item['Text'])

    # Join all extracted text lines into a single string
    extracted_text = '\n'.join(extracted_text)
    ingredients = [ingredient.strip() for ingredient in extracted_text.split(",")]
    return ingredients

def generate_summary(ingredients):
    title_template = PromptTemplate(
        input_variables=['ingredients'],
        template="""You are a health and nutrition specialist and I need you to analyse these ingredients and provide a summary about how healthy these are and what effects it can have on a person's health. Here are the ingredients: {ingredients} Provide a precise analysis on how healthy each ingredient is and then provide an overall summary of the healthiness of the ingredients."""
    )

    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='output')
    response = title_chain.invoke({'ingredients': ingredients})
    return response["output"]

@app.post("/extract_and_summarize")
async def extract_and_summarize(file: UploadFile = File(...)):
    # Read the uploaded image file as bytes
    image_bytes = await file.read()

    # Call OCR function to extract text from the image bytes
    ingredients = extract_text_from_image(image_bytes)

    # Call the generate_summary function with the extracted ingredients
    result = generate_summary(ingredients)

    return {
        "ingredients" : ingredients,
        "summary" : result
    }
