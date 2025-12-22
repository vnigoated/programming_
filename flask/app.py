from flask import Flask, request, template_rendered , jsonify

#create a application
app=Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Flask Application!"
@app.route('/form', methods=['GET','POST'])
def form():
    if request.method=='GET':
        name=request.form.get('form.html')

@app.route('/api', methods=['POST'])
def calculate_sum():
    data=request.get_json()
    a_val=dict(data)['a']
    b_val=dict(data)['b']
    return jsonify(int(a_val)+int(b_val))
        


#variable rules
@app.route('/success/<int:score>', methods=['GET'])
def success(score):
    return score

if __name__ == "__main__":
    app.run(debug=True)
    ##url routing

