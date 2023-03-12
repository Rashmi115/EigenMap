from flask import Flask, jsonify
import socket
from logisticRegression import *

app = Flask(__name__)

@app.route('/')
def Home():
    return "Flask application"

@app.route('/getip')
def GetIP():
    ip_address = socket.gethostbyname(socket.gethostname())
    ip = {'ip_address': ip_address}
    return jsonify(ip)

@app.route('/build-model')
def BuildModel():
    return jsonify(build_model())