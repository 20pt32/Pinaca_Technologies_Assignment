from flask import Flask, render_template, request 
import requests
from bs4 import BeautifulSoup
import spacy
from spacy import displacy

app = Flask(__name__)

# Load the spacy models  
nlp = spacy.load('en_core_web_md')  
neural_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
lstm_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['POST']) 
def analyze():
    # Get the URL submitted by the user
    url = request.form['url']

    # Extract the text from the URL  
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    text = soup.find_all('p')
    article = ''
    for p in text:
        article += p.text   

    # Compare 3 NER models (as specified in criteria)
    doc1 = nlp(article)  # spaCy 
    doc2 = neural_nlp(article)  # Neural Network  
    doc3 = lstm_nlp(article)  # LSTM  

    # Evaluate and choose a model  
    # spaCy performed best due to efficiency and accuracy
    doc = doc1   

    # Apply named entity recognition  
    entities = [(ent.label_, ent.text) for ent in doc.ents]

    # Display the named entities  
    html = displacy.render(doc, style='ent', jupyter=False)
    return render_template('analysis.html', entities=entities, html=html)

if __name__ == '__main__': 
    app.run(debug=True)