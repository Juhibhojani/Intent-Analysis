from flask import render_template
import pickle
from flask import Flask, request
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from statistics import mode

app = Flask(__name__)


def clean_text(text):
 # remove twitter account handle
 text = re.sub("@([A-Za-z0-9_]{1,15})", " ", text)

 # remove URL
 url_clean = re.compile(r"https://\S+|www\.\S+")
 text = url_clean.sub(r'', text)

 # remove noise
 text = re.sub(r'[^\x00-\x7F]+', ' ', text)

 # remove numbers
 text = re.sub(r'\d+', ' ', text)

 # remove hashtags
 text = re.sub(r'#\w+', ' ', text)

 # lower text
 text = text.strip().lower()

 # remove punctuation
 translator = str.maketrans(' ', ' ', string.punctuation)
 text = text.translate(translator)

 # remove stopwords
 stop_words = set(stopwords.words('english'))
 words = word_tokenize(text)
 filtered_words = [word for word in words if word not in stop_words]
 text = ' '.join(filtered_words)

 return text


def tokenize(text):
 tokens = word_tokenize(text)
 return tokens


lemmatizer = WordNetLemmatizer()
def lemming(content):
 lemmed_content = [lemmatizer.lemmatize(word) for word in content]
 return lemmed_content


def input_fun(text):
 return text

## Loading model files
#with open('rfl.pkl', 'rb') as rf_model_file:
 #rf = pickle.load(rf_model_file)

with open('svc.pkl', 'rb') as svc_model_file:
 svc = pickle.load(svc_model_file)

with open('lr.pkl', 'rb') as lr_model_file:
 logreg = pickle.load(lr_model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
 tfidf_vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
 le = pickle.load(encoder_file)


def preprocess_predict(text):
 text = clean_text(text)
 tokens = tokenize(text)
 l_tokens = lemming(tokens)
 vector = tfidf_vectorizer.transform(l_tokens)
 output = []
 #output.append(le.inverse_transform(rf.predict(vector))[0])
 output.append(le.inverse_transform(svc.predict(vector))[0])
 output.append(le.inverse_transform(logreg.predict(vector))[0])
 y_out = mode(output)
 return y_out


@app.route('/')
def home():
 return render_template('index.html')


@app.route('/predict', methods=['POST'])

def predict():
 text = str(request.form.values())
 output = preprocess_predict(text)
 return render_template('index.html', prediction=f"Intent : {output}")

if __name__ == '__main__':
 app.run(debug=True)
