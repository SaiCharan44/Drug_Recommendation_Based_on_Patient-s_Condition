from flask import Flask, render_template,request,redirect,session, url_for,flash
import os
import joblib
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from flask_pymongo import PyMongo

from bson import ObjectId

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

app = Flask(__name__)
app.config['MONGO_URI']='mongodb://localhost:27017/drugss'
app.config['SECRET_KEY'] = 'your_secret_key'
mongo=PyMongo(app)
MODEL_PATH = 'passmodel.pkl'
TOKENIZER_PATH ='tfidfvectorizer.pkl'
DATA_PATH ='data/drugsComTrain_raw.csv'
vectorizer = joblib.load(TOKENIZER_PATH)
model = joblib.load(MODEL_PATH)
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

@app.route('/')
def login():
    if 'username' in session:
        a=True
        return render_template('Main.html',a=a)
    else:
        a=False
        return render_template('Main.html',a=a)
@app.route("/book")
def book():
    return redirect("https://www.apollo247.com/specialties")


@app.route("/logout")
def logout():
	session.pop('username', None)
	return redirect('/')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if mongo.db.record.find_one({'username': username}):
            return 'Username already exists!'
        # Check if username already exists
        mongo.db.record.insert_one({'username': username, 'password': password})
    
    return render_template('login.html')
@app.route("/about")
def about():
    if 'username' in session:
        a=True
        return render_template('about.html',a=a)
    else:
        a=False
        return render_template('about.html',a=a)

@app.route('/index')
def index():
	if 'username' in session:	
		return render_template('home.html')
	else:
		return redirect('/')
    # Check if the user is logged in

@app.route('/register1')
def register1():
    return render_template('register.html')

@app.route('/login1')
def login1():
    return render_template('login.html')


@app.route('/predict1')
def predict1():
    if 'username' not in session:
        return render_template("login.html")
    else:
        return render_template("home.html")


@app.route('/history')
def history():
    mongodb_data = mongo.db.record.find_one({'username': session['username']})
    username=session['username']
    data_items=mongodb_data.get('data_items',[]) if mongodb_data else []
    return render_template('history.html',data_items=data_items,username=username)


@app.route('/login_validation', methods=['POST'])
def login_validation():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = mongo.db.record.find_one({'username': username, 'password': password})
        
        if user:
            session['username']=username
            a=True
            return render_template('Main.html',a=a)
        else:
            return render_template('login.html', message="Invalid username or password. Please try again.")
@app.route('/predict',methods=["GET","POST"])
# def predict():
# 	if request.method == 'POST':
# 		raw_text = request.form['rawtext']
# 		if raw_text != "":
# 			clean_text = cleanText(raw_text)
# 			clean_lst = [clean_text]
# 			tfidf_vect = vectorizer.transform(clean_lst)
# 			prediction = model.predict(tfidf_vect)
# 			predicted_cond = prediction[0]
# 			df = pd.read_csv(DATA_PATH)
# 			top_drugs = top_drugs_extractor(predicted_cond,df)
# 			return render_template('predict.html',rawtext=raw_text,rawtext1=clean_text,result=predicted_cond,top_drugs=top_drugs)
# 		else:
# 			 raw_text ="There is no text to select"
def predict():
    if request.method=='POST':
        raw_text=request.form['rawtext']
        if raw_text!="":
            clean_text=cleanText(raw_text)
            clean_lst=[clean_text]
            tfidf_vect=vectorizer.transform(clean_lst)
            prediction=model.predict(tfidf_vect)
            predicted_cond=prediction[0]
            df=pd.read_csv(DATA_PATH)
            top_drugs=top_drugs_extractor(predicted_cond,df)
            data={'rawtext':raw_text,'cleantext':clean_text,'result':predicted_cond,'top_drugs':top_drugs}
            user_data = mongo.db.record.find_one({'username': session['username']})
            if 'username' in session:
                data={
                    'rawtext':raw_text,
                    'cleantext':clean_text,
                    'result':predicted_cond,
                    'top_drugs':top_drugs
                    
                }
            if user_data and 'data_items' in user_data and isinstance(user_data['data_items'],list):
                result=mongo.db.record.update_one({'username':session['username']},{'$addToSet':{'data_items':data}})
            else:
                result=mongo.db.record.update_one({'username':session['username']},{'$set':{'data_items':[data]}})
            # if user_data and 'data' in user_data:
            #     if isinstance(user_data['data'], list):
            #         mongo.db.record.update_one({'username': session['username']}, {'$addToSet': {'data': data}})
            #     else:
            #         mongo.db.record.update_one({'username': session['username']}, {'$set': {'data': [user_data['data'], data]}})
            # else:
            #     mongo.db.record.update_one({'username': session['username']}, {'$set': {'data': data}})


            return render_template('predict.html',rawtext=raw_text,cleantext=clean_text,result=predicted_cond,top_drugs=top_drugs)
        else:
            raw_text ="There is no text to select"
   


def cleanText(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))


def top_drugs_extractor(condition,df):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst







if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)