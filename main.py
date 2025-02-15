# /// script
# requires-python = ">=3.8"  # Adjust based on your minimum required Python version
# dependencies = [
#     "fastapi>=0.112.0,<0.113.0",
#     "uvicorn[standard]",  # Uvicorn with recommended dependencies
#     "requests",
#     "openai",
#     "Pillow",
#     "numpy",
#     "pytesseract",
#     "fastapi",
#     "datetime",
#     "pathlib",
# ]
# ///

# app.py (Merged with tasksA.py and tasksB.py)
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
import os
import re
import httpx
import json
import sqlite3
import subprocess
from dateutil.parser import parse
from datetime import datetime
from pathlib import Path
from scipy.spatial.distance import cosine
import base64
from PIL import Image
import markdown
import duckdb

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# OpenAI API configuration
openai_api_chat = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"  # for testing
openai_api_key = os.getenv("AIPROXY_TOKEN")

headers = {
    "Authorization": f"Bearer {openai_api_key}",
    "Content-Type": "application/json",
}

# Function definitions for LLM
function_definitions_llm = [
    {
        "name": "A1",
        "description": "Run a Python script from a given URL, passing an email as the argument.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "pattern": r"[\w\.-]+@[\w\.-]+\.\w+"}
            },
            "required": ["email"]
        }
    },
    {
        "name": "A2",
        "description": "Format a markdown file using a specified version of Prettier.",
        "parameters": {
            "type": "object",
            "properties": {
                "prettier_version": {"type": "string", "pattern": r"prettier@\d+\.\d+\.\d+"},
                "filename": {"type": "string", "pattern": r".*/(.*\.md)"}
            },
            "required": ["prettier_version", "filename"]
        }
    },
    {
        "name": "A3",
        "description": "Count the number of occurrences of a specific weekday in a date file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r"/data/.*dates.*\.txt"},
                "targetfile": {"type": "string", "pattern": r"/data/.*/(.*\.txt)"},
                "weekday": {"type": "integer", "pattern": r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"}
            },
            "required": ["filename", "targetfile", "weekday"]
        }
    },
    {
        "name": "A4",
        "description": "Sort a JSON contacts file and save the sorted version to a target file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r".*/(.*\.json)"},
                "targetfile": {"type": "string", "pattern": r".*/(.*\.json)"}
            },
            "required": ["filename", "targetfile"]
        }
    },
    {
        "name": "A5",
        "description": "Retrieve the most recent log files from a directory and save their content to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "log_dir_path": {"type": "string", "pattern": r".*/logs", "default": "/data/logs"},
                "output_file_path": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/logs-recent.txt"},
                "num_files": {"type": "integer", "minimum": 1, "default": 10}
            },
            "required": ["log_dir_path", "output_file_path", "num_files"]
        }
    },
    {
        "name": "A6",
        "description": "Generate an index of documents from a directory and save it as a JSON file.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_dir_path": {"type": "string", "pattern": r".*/docs", "default": "/data/docs"},
                "output_file_path": {"type": "string", "pattern": r".*/(.*\.json)", "default": "/data/docs/index.json"}
            },
            "required": ["doc_dir_path", "output_file_path"]
        }
    },
    {
        "name": "A7",
        "description": "Extract the sender's email address from a text file and save it to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/email.txt"},
                "output_file": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/email-sender.txt"}
            },
            "required": ["filename", "output_file"]
        }
    },
    {
        "name": "A8",
        "description": "Generate an image representation of credit card details from a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/credit-card.txt"},
                "image_path": {"type": "string", "pattern": r".*/(.*\.png)", "default": "/data/credit-card.png"}
            },
            "required": ["filename", "image_path"]
        }
    },
    {
        "name": "A9",
        "description": "Find similar comments from a text file and save them to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/comments.txt"},
                "output_filename": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/comments-similar.txt"}
            },
            "required": ["filename", "output_filename"]
        }
    },
    {
        "name": "A10",
        "description": "Identify high-value (gold) ticket sales from a database and save them to a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r".*/(.*\.db)", "default": "/data/ticket-sales.db"},
                "output_filename": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/ticket-sales-gold.txt"},
                "query": {"type": "string", "pattern": "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"}
            },
            "required": ["filename", "output_filename", "query"]
        }
    },
    {
        "name": "B12",
        "description": "Check if filepath starts with /data",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "pattern": r"^/data/.*"}
            },
            "required": ["filepath"]
        }
    },
    {
        "name": "B3",
        "description": "Download content from a URL and save it to the specified path.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "pattern": r"https?://.*"},
                "save_path": {"type": "string", "pattern": r".*/.*"}
            },
            "required": ["url", "save_path"]
        }
    },
    {
        "name": "B5",
        "description": "Execute a SQL query on a specified database file and save the result to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "db_path": {"type": "string", "pattern": r".*/(.*\.db)"},
                "query": {"type": "string"},
                "output_filename": {"type": "string", "pattern": r".*/(.*\.txt)"}
            },
            "required": ["db_path", "query", "output_filename"]
        }
    },
    {
        "name": "B6",
        "description": "Fetch content from a URL and save it to the specified output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "pattern": r"https?://.*"},
                "output_filename": {"type": "string", "pattern": r".*/.*"}
            },
            "required": ["url", "output_filename"]
        }
    },
    {
        "name": "B7",
        "description": "Process an image by optionally resizing it and saving the result to an output path.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "pattern": r".*/(.*\.(jpg|jpeg|png|gif|bmp))"},
                "output_path": {"type": "string", "pattern": r".*/.*"},
                "resize": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 2}
            },
            "required": ["image_path", "output_path"]
        }
    },
    {
        "name": "B9",
        "description": "Convert a Markdown file to another format and save the result to the specified output path.",
        "parameters": {
            "type": "object",
            "properties": {
                "md_path": {"type": "string", "pattern": r".*/(.*\.md)"},
                "output_path": {"type": "string", "pattern": r".*/.*"}
            },
            "required": ["md_path", "output_path"]
        }
    }
]

# Task A Functions
def A1(email="xxxxxxxxx@ds.study.iitm.ac.in"):
    try:
        process = subprocess.Popen(
            ["uv", "run", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", email],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error: {stderr}")
        return stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")

def A2(prettier_version="prettier@3.4.2", filename="/data/format.md"):
    command = [r"C:\Program Files\nodejs\npx.cmd", prettier_version, "--write", filename]
    try:
        subprocess.run(command, check=True)
        print("Prettier executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def A3(filename='/data/dates.txt', targetfile='/data/dates-wednesdays.txt', weekday=2):
    input_file = filename
    output_file = targetfile
    weekday_count = 0

    with open(input_file, 'r') as file:
        weekday_count = sum(1 for date in file if parse(date).weekday() == int(weekday)-1)

    with open(output_file, 'w') as file:
        file.write(str(weekday_count))

def A4(filename="/data/contacts.json", targetfile="/data/contacts-sorted.json"):
    with open(filename, 'r') as file:
        contacts = json.load(file)
    sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))
    with open(targetfile, 'w') as file:
        json.dump(sorted_contacts, file, indent=4)

def A5(log_dir_path='/data/logs', output_file_path='/data/logs-recent.txt', num_files=10):
    log_dir = Path(log_dir_path)
    output_file = Path(output_file_path)
    log_files = sorted(log_dir.glob('*.log'), key=os.path.getmtime, reverse=True)[:num_files]
    with output_file.open('w') as f_out:
        for log_file in log_files:
            with log_file.open('r') as f_in:
                first_line = f_in.readline().strip()
                f_out.write(f"{first_line}\n")

def A6(doc_dir_path='/data/docs', output_file_path='/data/docs/index.json'):
    docs_dir = os.path.abspath(doc_dir_path)
    output_file = os.path.abspath(output_file_path)
    index_data = {}

    # Walk through all files in the docs directory
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('# '):
                            title = line[2:].strip()
                            relative_path = os.path.relpath(file_path, docs_dir).replace('\\', '/')
                            index_data[relative_path] = title
                            break

    # Write the index data to index.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4)

    print("Index Data:", index_data)  # Debugging
    
def A7(filename='/data/email.txt', output_file='/data/email-sender.txt'):
    with open(filename, 'r') as file:
        email_content = file.readlines()
    sender_email = "sujay@gmail.com"
    for line in email_content:
        if "From" == line[:4]:
            sender_email = (line.strip().split(" ")[-1]).replace("<", "").replace(">", "")
            break
    with open(output_file, 'w') as file:
        file.write(sender_email)

def png_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def A8(filename='/data/credit_card.txt', image_path='/data/credit_card.png'):
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "There is 8 or more digit number is there in this image, with space after every 4 digit, only extract the those digit number without spaces and return just the number without any other characters"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{png_to_base64(image_path)}"
                        }
                    }
                ]
            }
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}"
    }
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                             headers=headers, data=json.dumps(body))
    card_number = response.json()['choices'][0]['message']['content'].replace(" ", "")
    with open(filename, 'w') as file:
        file.write(card_number)

def get_embedding(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": [text]
    }
    try:
        response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/embeddings", headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e}")
        print(f"Response: {response.text}")
        return None

def A9(filename='/data/comments.txt', output_filename='/data/comments-similar.txt'):
    # Read comments
    with open(filename, 'r') as f:
        comments = [line.strip() for line in f.readlines()]

    # Get embeddings for all comments
    embeddings = [get_embedding(comment) for comment in comments]

    # Find the most similar pair
    min_distance = float('inf')
    most_similar = (None, None)

    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            distance = cosine(embeddings[i], embeddings[j])
            if distance < min_distance:
                min_distance = distance
                most_similar = (comments[i], comments[j])

    # Write the most similar pair to file
    with open(output_filename, 'w') as f:
        f.write(most_similar[0] + '\n')
        f.write(most_similar[1] + '\n')

    print("Most Similar Pair:", most_similar)  # Debugging
def A10(filename='/data/ticket-sales.db', output_filename='/data/ticket-sales-gold.txt', query="SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"):
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()
    cursor.execute(query)
    total_sales = cursor.fetchone()[0]
    total_sales = total_sales if total_sales else 0
    with open(output_filename, 'w') as file:
        file.write(str(total_sales))
    conn.close()

# Task B Functions
def B12(filepath):
    return filepath.startswith('/data')

def B3(url, save_path):
    if not B12(save_path):
        return None
    response = requests.get(url)
    with open(save_path, 'w') as file:
        file.write(response.text)

def B5(db_path, query, output_filename):
    if not B12(db_path):
        return None
    conn = sqlite3.connect(db_path) if db_path.endswith('.db') else duckdb.connect(db_path)
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    conn.close()
    with open(output_filename, 'w') as file:
        file.write(str(result))
    return result

def B6(url, output_filename):
    result = requests.get(url).text
    with open(output_filename, 'w') as file:
        file.write(str(result))

def B7(image_path, output_path, resize=None):
    if not B12(image_path) or not B12(output_path):
        return None
    img = Image.open(image_path)
    if resize:
        img = img.resize(resize)
    img.save(output_path)

def B9(md_path, output_path):
    if not B12(md_path) or not B12(output_path):
        return None
    with open(md_path, 'r') as file:
        html = markdown.markdown(file.read())
    with open(output_path, 'w') as file:
        file.write(html)

# FastAPI Endpoints
@app.get("/ask")
def ask(prompt: str):
    result = get_completions(prompt)
    return result

@app.post("/run")
async def run_task(task: str):
    try:
        response = get_completions(task)
        task_code = response['name']
        arguments = response['arguments']
        if task_code == "A1":
            A1(**json.loads(arguments))
        elif task_code == "A2":
            A2(**json.loads(arguments))
        elif task_code == "A3":
            A3(**json.loads(arguments))
        elif task_code == "A4":
            A4(**json.loads(arguments))
        elif task_code == "A5":
            A5(**json.loads(arguments))
        elif task_code == "A6":
            A6(**json.loads(arguments))
        elif task_code == "A7":
            A7(**json.loads(arguments))
        elif task_code == "A8":
            A8(**json.loads(arguments))
        elif task_code == "A9":
            A9(**json.loads(arguments))
        elif task_code == "A10":
            A10(**json.loads(arguments))
        elif task_code == "B12":
            B12(**json.loads(arguments))
        elif task_code == "B3":
            B3(**json.loads(arguments))
        elif task_code == "B5":
            B5(**json.loads(arguments))
        elif task_code == "B6":
            B6(**json.loads(arguments))
        elif task_code == "B7":
            B7(**json.loads(arguments))
        elif task_code == "B9":
            B9(**json.loads(arguments))
        return {"message": f"{task_code} Task '{task}' executed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="File path to read")):
    try:
        with open(path, "r") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to get completions from OpenAI
def get_completions(prompt: str):
    with httpx.Client(timeout=20) as client:
        response = client.post(
            f"{openai_api_chat}",
            headers=headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a function classifier that extracts structured parameters from queries."},
                    {"role": "user", "content": prompt}
                ],
                "tools": [{"type": "function", "function": function} for function in function_definitions_llm],
                "tool_choice": "auto"
            },
        )
    return response.json()["choices"][0]["message"]["tool_calls"][0]["function"]

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)