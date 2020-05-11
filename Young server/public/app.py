import pandas as pd
from flask import Flask, jsonify, render_template,redirect,request

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
        ma1 = request.form['ma1']
        ma2 = request.form['ma2']
        img = './static/figure.png'

        # Parameters can now be passed through for calculations

      
        return render_template("submit.html",ma1=ma1,ma2=ma2,ticker=ticker,img=img)   
    else:
        return render_template("submit.html")



if __name__ == "__main__":
    app.run(debug=True)