from flask import Flask,render_template
app=Flask(__name__)

@app.route("/")
def hello():
   return render_template('page1.html')

@app.route("/authenticationpage")
def about():
   return render_template('page2.html')   


if __name__=='__main__':
    app.run(debug=True)  