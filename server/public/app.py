import pandas as pd
from flask import Flask, jsonify, render_template,redirect,request
import math 
import numpy as np
import pandas as pd
import datetime
from pandas import Series, DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from analyze1 import generatePlot
from predict import forecast

app=Flask(__name__)

@app.route("/",methods = ["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/submit",methods=["POST","GET"])
def submit():
    if request.method == 'POST':
        req = request
        print(req.form)
        ticker = request.form['ticker']
        ma1 = int(request.form['ma1'])
        ma2 = int(request.form['ma2'])
        
        # Parameters can now be passed through for calculations
        forecast(ma1,ma2,ticker)
        img = './static/predict.png'
        return render_template("submit.html",ma1=ma1,ma2=ma2,ticker=ticker,img=img)   
    else:
        return render_template("submit.html")

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == "__main__":
    app.run(debug=True)