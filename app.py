from flask import Flask, jsonify
import socket
from logisticRegression import *

app = Flask(__name__)

model = ""

@app.route('/')
def Home():
    return jsonify("Flask application")

@app.route('/getip')
def GetIP():
    ip_address = socket.gethostbyname(socket.gethostname())
    ip = {'ip_address': ip_address}
    return jsonify(ip)

@app.route('/build-model')
def BuildModel():
    global model 
    model, measures = build_model()
    return jsonify(measures)
