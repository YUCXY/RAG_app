from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, Response
from werkzeug.utils import secure_filename

import time
import os
import re
import urllib.request
import requests
from requests.exceptions import RequestException
import numpy as np
from pypdf import PdfReader
from bs4 import BeautifulSoup
import validators

from rag import get_response, tokenize, get_doc_embeddings


UPLOAD_FOLDER = './documents'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

doc_path = ""
doc = ""
doc_embeddings = {}
doc_tokens = {}
output = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file):
    """
    Helper function which extracts all text from PDF file and saves each page as an item in document variable.
    """
    reader = PdfReader(file)
    document = ""
    for page in reader.pages:
        txt = page.extract_text()
        document += txt
    
    tokens = tokenize(document, 100)
    embeddings = get_doc_embeddings(tokens)
    return document, tokens, embeddings

def extract_text_from_html(file):
    """
    Helper function which extracts all text from html file and saves as an item in document variable.
    """
    document = ""
    with urllib.request.urlopen(file) as page:
        soup = BeautifulSoup(page.read())
        document += soup.get_text()
    
    tokens = tokenize(document, 100)
    embeddings = get_doc_embeddings(tokens)
    return document, tokens, embeddings

@app.route('/')
def home():
    return render_template('index.html', message='RAG Chat')

@app.route('/upload_pdf')
def upload_pdf():
    return render_template('upload_pdf.html') # form.html specifies action submit (POST request)

@app.route('/select_url')
def select_url():
    return render_template('select_url.html')

@app.route('/select', methods=['POST'])
def select():
    global doc, doc_tokens, doc_embeddings
    url = request.form['url']

    if not validators.url(url):
        flash("Invalid website. Please enter a valid url.")
        return redirect('/select_url')
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            url = request.form['url']
            doc, doc_tokens, doc_embeddings = extract_text_from_html(url)
            return redirect(url_for('chat'), code=307)
        else:
            flash("Invalid website. Please enter a valid url.")
            return redirect('/select_url')
    except ConnectionError as e:
        flash("Invalid website. Please enter a valid url.")
        return redirect('/select_url')
    except RequestException as e:
        flash("Invalid website. Please enter a valid url.")
        return redirect('/select_url')

@app.route('/submit', methods=['POST'])
def submit():
    """
    Handle file upload and process the uploaded PDF file.
    """
    global doc, doc_tokens, doc_embeddings

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('Invalid file selected. Please select a PDF file.')
        return redirect(request.url)

    try:
        doc, doc_tokens, doc_embeddings = extract_text_from_pdf(file)
        return redirect(url_for('chat'), code=307)  # Redirect to chat with POST method
    except Exception as e:
        flash(f"An error occurred while processing the file: {str(e)}")
        return redirect('/upload_pdf')

@app.route('/display', methods=['POST'])
def display_file():
    """
    Display selected document.
    """
    return render_template('display.html', content=doc)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat with DeepSeek-v3 for RAG chat session about uploaded document or website.
    """
    return render_template('chat.html', content=[])

@app.route('/get_query', methods=['POST'])
def get_query():
    """
    Get chat queries.
    """
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Invalid query. Ask a question to understand more about the document."})
    
    response = get_response(query, doc, doc_tokens, doc_embeddings)
    words = response.split()
    
    def generate():
        for word in words:
            yield word + " "
            time.sleep(0.15)

    return Response(generate(), content_type="text/event-stream")

if __name__ == "__main__":
    app.secret_key='12345test'
    app.run(debug=True)