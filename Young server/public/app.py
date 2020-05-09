import pandas as pd
from flask import Flask, jsonify, render_template,redirect,request

app=Flask(__name__)

@app.route("/",methods = ["GET","POST"])
def index():
    # print('hiii')
    # print('method:  ',request.method)
    if request.method == "POST":
        print('hey there')
        req = request
        print(req.form)
        # print(username)
        # return redirect(request.url)

    return render_template("index.html")

@app.route("/submit",methods=["POST","GET"])
def submit():
    if request.method == 'POST':
        req = request
        print(req.form)
        return render_template("submit.html")   
    else:
        return render_template("submit.html")



if __name__ == "__main__":
    app.run(debug=True)