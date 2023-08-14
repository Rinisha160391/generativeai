from flask import Flask,request, render_template, jsonify, Response
from werkzeug.utils import secure_filename
from pdf_processor import pdf_processor

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        if 'file' not in request.files:
            return jsonify({"error":"No fie was sent as part of the request"}),400

        file=request.files['file']
        if file.filename=='':
            return jsonify({"error":"No file was selected"}),400
        
        filename=secure_filename(file.filename)
        file.save(filename)
        
        question=request.form['question']
        response=pdf_processor(filename,question)
        
        return render_template('upload.html', response=response)
    return render_template('upload.html')
            
if __name__=='__main__':
    app.run()

