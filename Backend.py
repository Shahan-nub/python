from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS  # Import CORS
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    ans = []
    for char in text:
        if(char.isalnum()):
            y.append(char)
    for word in y:
        if(word not in stopwords.words('english')):
            ans.append(word)
    y.clear()

    ps = PorterStemmer()
    for word in ans:
        y.append(ps.stem(word))
    return " ".join(y)


vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('Spam_mail_Model.pkl','rb'))

app = Flask("spam-detector")
CORS(app)  # Enable CORS for all routes (for development)


# try:
#     vectorizer = pickle.load(open('vectorizer.pkl','rb'))
#     model = pickle.load(open('Spam_mail_Model.pkl','rb'))
# except FileNotFoundError:
#     print("Model file not found. Make sure 'iris_model.pkl' is in the same directory.")
#     exit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Data :{ ",data," }")
        transformed_text = text_transform(data)
        vector_input = vectorizer.transform([transformed_text])

        result = model.predict(vector_input)[0]
        print("result : ",result)
        if result == 1:
            return jsonify({'prediction': "SPAM"})
        if result == 0:
            return jsonify({'prediction':"HAM"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # host 0.0.0.0 for making it accessible over the network