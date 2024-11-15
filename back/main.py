from flask import Flask, request, jsonify
from compiler import SQLParameterExtractor

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/search", methods=["POST"])
def search():
    try:
        if not request.is_json:
            return jsonify({"error": "Content type must be application/json"}), 400
        
        body = request.json
        if 'query' not in body:
            return jsonify({"error": "Missing 'query' field"}), 400
            
        compiler = SQLParameterExtractor()
        for query in body['query'].split(';'):
            query = query.strip()
            if not query:
                continue
                
            operation, params = compiler.process_query(query)
            if operation == "CREATE_TABLE":
                print("CREATE TABLE", params)
            elif operation == "CREATE_INDEX":
                print("CREATE INDEX", params)
            elif operation == "SELECT":
                print("SELECT", params)
            else:
                print("UNKNOWN", params)
        return jsonify({"status": "ok"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    """
    Para correr el servidor Flask:
    . .venv/bin/activate
    flask --app main --debug run
    """
    app.run(debug=True)
