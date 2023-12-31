from flask import Flask

app = Flask(__name__)


@app.route('/start')
def start():
    return 'start'


@app.route('/select/<name>')
def select(name):
    return 'hi %s' % name


@app.route('/')
def hello_world():
    return '<h1>Hello World!</h1><input type="textbox"/>'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
