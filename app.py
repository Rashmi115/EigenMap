from flask import Flask, jsonify
import socket
from logisticRegression import *
from cnn import *

app = Flask(__name__)

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
    model, measures = build_model()
    return jsonify(measures)

@app.route('/build-cnn')
def BuildCNN():
    cnn_model, measures = build_cnn()
    return jsonify(measures)
