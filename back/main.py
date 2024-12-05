from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import NamedTupleCursor
from compiler import CompilerExecute
from datetime import datetime
import hashlib
import os
from pymongo import MongoClient
from parser import process_query, ManageExceptions

import json
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/csv_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


DATABASE_CONFIG_ADMIN = {
    'host': '127.0.0.1',
    'database': 'informationretrieval',
    'user': 'postgres',
    'password': '07283658',
    'port': 5432
}

DATABASE_WORK = {
    'host': '127.0.0.1',
    'database': 'workiretrieval',
    'user': 'admin_if',
    'password': 'administrator@if$',
    'port': 5432
}

client = MongoClient('mongodb://localhost:27017')
db_mongo = client.userfolders


def db_connection(database):
    conection = psycopg2.connect(**database)
    return conection

def generate_folder_path(input_string):
    hash_object = hashlib.sha256(input_string.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    folder_path = str(hash_hex)
    return folder_path

def save_userFolder(path, schema_user):
    try:
        db_mongo.iretrieval.insert_one({ 
            '_id': path, 
            'schema': schema_user,
            'folder' :{
                'tables':[],
                'files':[],
                'dataset':[]
            }
        })
        return True
    except Exception as e:
        print("Error:", e)
        return False
        
def search_userFolder(path):
    try:
        document = db_mongo.iretrieval.find_one({'_id': str(path)})
        print(document)
        return document
    except Exception as e:
        print("No encontrado:", e)
        return None    

def search_indexes(path):
    try:
        document = db_mongo.iretrieval.find({'_id': str(path)},{'folder':1})
        resultado = []
        list_gin = []
        list_own = []

        for doc in document:
            folder = doc.get("folder", {})

            tablas = folder.get("tables", [])
            for tabla in tablas:
                indices = tabla.get("index", [])
                for indice in indices:
                    if indice.get("by")=="GIN":
                        list_gin.append({
                            "name": tabla.get("tableName"),
                            "columns": indice.get("columns"),
                            "index": indice.get("name")
                            
                        })
                    else:
                        list_own.append({
                            "name": tabla.get("tableName"),
                            "columns": indice.get("columns"),
                            "index": indice.get("name"),
                            "language":indice.get("language")
                        })

            datasets = folder.get("dataset", [])
            for dataset in datasets:
                if dataset["hasIndex"]:
                    for idx in dataset.get("index",[]):
                        list_own.append({
                            "name": dataset.get("csv"),
                            "columns": idx.get("columns"),
                            "language":idx.get("language")
                        })
        resultado = {"gin":list_gin,"own":list_own}
        print("Resultado: \n", resultado)
        return resultado
    except Exception as e:
        print("No encontrado:", e)
        return None

def search_File(path, file):
    try:
        document = db_mongo.iretrieval.find_one({'_id': str(path),'folder.files.filename':file})
        return document
    except Exception as e:
        print("No encontrado:", e)
        return None    

def search_userTable(path, tablename):
    try:
        table = db_mongo.iretrieval.find_one(
            {
                '_id': path,
                'folder.tables.tableName': tablename
            },
            {
                '_id': 0,
                'folder.tables.$': 1
            }
        )
        if table != None: 
            return table['folder']['tables'][0]['csv']
        else: 
            return False
    except Exception as e:
        print("No encontrado:", e)
        raise ManageExceptions("Error buscando tabla", e)


def get_properties(indexes, type_index, dataset):

    if type_index not in indexes:
        print(f"El type_index '{type_index}' no se encuentra en los índices disponibles.")
        return None
    for ds in indexes[type_index]:
        if ds.get('name') == dataset:
            columns = ds.get('columns')
            index = ds.get('index')
            if ds.get('language'):
                language = ds.get('language')
                return {'columns': columns, 'index': index, 'language':language}
            return {'columns': columns, 'index': index}

    print(f"El dataset '{dataset}' no se encuentra bajo el type_index '{type_index}'.")
    return None
    
def verifify_exist_index(path,type_dataset, name_dataset, method):
    try:
        if type_dataset == "table":
            table = db_mongo.iretrieval.find_one(
                {
                    '_id': path,
                    'folder.tables.tableName': name_dataset,
                    'folder.tables.hasIndex': True
                },
                {
                    '_id': 0,
                    'folder.tables.$': 1
                }
            )
            if table!= None:
                for idx in table['folder']['tables'][0]['index']:
                    if(idx['by']==method):
                        return True
            return False

        else:
            dataset = db_mongo.iretrieval.find_one(
                {
                    '_id': path,
                    'folder.dataset.csv': name_dataset,
                    'folder.dataset.hasIndex': True
                },
                {
                    '_id': 0,
                    'folder.dataset.$': 1
                }
            )
            print("Aqui data",dataset)

            return True if dataset!= None else False                   
            

    except Exception as e:
        print("No encontrado:", e)
        return None


def add_file_to_user(username, filename, typeFile, url):
    try:
        id_user = generate_folder_path(username)
        print("id_user: ", id_user)
        result = db_mongo.iretrieval.update_one(
            {'_id': id_user},{'$push': {'folder.files': {'type':typeFile,'filename':filename, 'url':url}}}
        )
        return True
    except Exception as e:
        print(f"Error al crear el indice: {e}")
        return e

def add_table_to_user(username, table, columns, csv=""):
    try: 
        col_retrieval = []
        for col in columns:
            col_retrieval.append(col['name'])
        id_user = generate_folder_path(username)
        print("id_user: ", id_user)
        result = db_mongo.iretrieval.update_one(
            {'_id': id_user},{'$push': {'folder.tables': {'tableName':table,'columns':col_retrieval, 'hasIndex':False, 'csv':csv}}}
        )
        return True
    except Exception as e:
        print(f"Error al crear el indice: {e}")
        return e

def add_dataset_user(username, file):
    try: 
        id_user = generate_folder_path(username)
        result = db_mongo.iretrieval.update_one(
            {'_id': id_user},{'$push': {'folder.dataset': {'csv':file,'hasIndex':False}}}
        )
        return True
    except Exception as e:
        print(f"Error al crear el indice: {e}")
        return e


def add_index_to_table(username, table_name, index_name, method, column, language):
    try:
        id_user = generate_folder_path(username)
        print("id_user: ", id_user)
        result = db_mongo.iretrieval.update_one(
            {
                '_id': id_user,
                'folder.tables.tableName': table_name
            },
            {
                '$set': {
                    'folder.tables.$.hasIndex': True
                },
                '$push': {
                    'folder.tables.$.index': {"name":index_name, "columns":column, "by":method, "language":language}
                }
            }
        )
        if result.modified_count > 0:
            print("La tabla se actualizó correctamente.")
            return True
        else:
            print("No se encontró la tabla o no se realizó ninguna actualización.")
            return False
    except Exception as e:
        print(f"Error al crear el indice: {e}")
        return e

def add_index_to_dataset(username, dataset , index_name, column, language):
    try:
        id_user = generate_folder_path(username)
        print("id_user: ", id_user)
        print(username," ----" ,dataset," ----", index_name," ----", column)
        result = db_mongo.iretrieval.update_one(
            {
                '_id': id_user,
                'folder.dataset.csv': dataset
            },
            {
                '$set': {
                    'folder.dataset.$.hasIndex': True
                },
                '$push': {
                    'folder.dataset.$.index': {"name":index_name, "columns":column, "language":language}
                }
            }
        )
        if result.modified_count > 0:
            print("La tabla se actualizó correctamente.")
            return True
        else:
            print("No se encontró la tabla o no se realizó ninguna actualización.")
            return False
    except Exception as e:
        print(f"Error al crear el indice: {e}")
        return e

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/register", methods=["POST"])
def register():
    try:
        if not request.is_json:
            print("Error: Content type is not JSON")
            return jsonify({"error": "Content type must be application/json"}), 400

        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        
        conection_admin = db_connection(DATABASE_CONFIG_ADMIN)
        cursor = conection_admin.cursor(cursor_factory=NamedTupleCursor)        
        cursor.execute("SELECT username FROM users WHERE username = %s;", (username,))
        usuario_existente = cursor.fetchone()

        if usuario_existente:
            cursor.close()
            conection_admin.close()
            print("Username already exists")
            return jsonify({"message": "Username already exists"}), 401

        path = generate_folder_path(username)
        date_reg = datetime.today()
        cursor.execute(
            "INSERT INTO users (username, password, data_reg, folder_path) VALUES (%s, %s, %s, %s);",
            (username, password, date_reg, path)
        )
    
        conection_work = db_connection(DATABASE_WORK)
        cursor_work = conection_work.cursor()
        cursor_work.execute(f"CREATE SCHEMA user_{username};")
        
        
        if save_userFolder(path, "user_%s"%(username)):
            print("Usuario registrado correctamente")
            conection_work.commit()
            conection_admin.commit()
        else:
            return jsonify({"message": "Intentelo nuevamente"}), 400

        cursor.close()
        cursor_work.close()
        conection_work.close()
        conection_admin.close()

        return jsonify({"message": "User registered successfully"}), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/login",  methods=["POST"])
def login():
    try:
        if not request.is_json:
            print("Error: Content type is not JSON")
            return jsonify({"message": "Content type must be application/json"}), 400

        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        conection = db_connection(DATABASE_CONFIG_ADMIN)
        cursor = conection.cursor(cursor_factory=NamedTupleCursor)
        cursor.execute("SELECT * FROM users WHERE username = %s;", (username,))
        user_reg = cursor.fetchone()
        print(user_reg)

        if not user_reg:
            return jsonify({"message": "User not exits"}), 401


        elif user_reg.password == password:
            document_user = search_userFolder(user_reg.folder_path)
            cursor.close()
            conection.close()
            return jsonify({
                "message": "Login successful",
                "folder" : document_user['folder'],
                "indexes": search_indexes(user_reg.folder_path)
            }), 200

        elif username == "admin" and password == "123":

            
            folder = {
                "_id":"admin",
                "tables":[
                    {
                        "tableName":"table_1",
                        "columns":["id","name","text"],
                        "hasIdex":True, 
                        "index":[
                            {"name":"index_test", "by":"postgres"},
                            {"name":"index_own", "by":"own"}
                        ],
                    },
                    {
                        "tableName":"table_2",
                        "columns":["id_2","name_2","text_2"],
                        "hasIdex":False, 
                        "index":[
                            {"name":"index_test", "by":"postgres"},
                            {"name":"index_own", "by":"own"}
                        ],
                        "csv":""
                    }
                ],
                "files":[
                    {
                        "type":"img",
                        "fileName":"img_src.png"
                    },
                    {
                        "type":"csv",
                        "fileName":"data.csv"
                    }
                ],
                "dataset":[
                    "spotify-songs", "news-uk"
                ]
            }

            return jsonify({
                "message": "Login successful",
                "scripts":script_notes,
                "folder":folder
            }), 200
        else:
            return jsonify({"message": "Invalid credentials"}), 401
    except Exception as e:
        print("Error:", e)
        return jsonify({"message": str(e)}), 500

@app.route("/saveFile",  methods=["POST"])
def saveFile():
    try:
        if not request.is_json:
            print("Error: Content type is not JSON")
            return jsonify({"message": "Content type must be application/json"}), 400

        data = request.get_json()
        print("SaveFile: ",data)
        username = data.get("username")
        filename = data.get("name")
        typeFile = data.get("type")
        url = data.get("url")

        add_file_to_user(username, filename, typeFile, url)
        return jsonify({
                "message": "Archivo subido correctamente",
                "folder":search_userFolder(generate_folder_path(username))['folder']
            }), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({"message": str(e)}), 500

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    print("Iniciando")
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    print("File recuperado")
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    print("verificando csv")
    if not file.filename.endswith('.csv'):
        return jsonify({'message': 'File is not a CSV'}), 400

    try:
        print("Iniciando guardado")
        path = app.config['UPLOAD_FOLDER']+"/"+str(generate_folder_path(request.form['username']))
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, file.filename)
        print(file_path)
        file.save(file_path)
        add_file_to_user(request.form['username'], file.filename, 'text/csv', file.filename)

        return jsonify({
                "message": "Archivo subido correctamente",
                "folder":search_userFolder(generate_folder_path(request.form['username']))['folder']
            }), 200
        
    except Exception as e:
        print("Error:", e)
        return jsonify({'message': f'Error saving file: {str(e)}'}), 500

@app.route("/run", methods=["POST"])
def search():
    try:
        if not request.is_json:
            return jsonify({"message": "Content type must be application/json"}), 400
        
        body = request.get_json()
        if 'type' not in body:
            return jsonify({"message": "Missing 'type' field"}), 400
        
        
        
        typeQuery = body['type']
        textQuery = body['inputText']
        username = body['username']
        result = ""
        images = []
        compiler = CompilerExecute()
        if typeQuery=="create":
            queryString = process_query(textQuery)
            queryJSON = json.loads(queryString)
            for query in queryJSON:
                print("Query",query)
                
                if query["action"] == 'CREATE DATASET':
                    if search_File(generate_folder_path(username),query['source_file']):
                        add_dataset_user(username, query['source_file'])
                    else:
                        raise ManageExceptions("Ups! Archivo no encontrado")
                
                elif query["action"] == 'CREATE TABLE':
                    path_user = str(generate_folder_path(username))
                    path = app.config['UPLOAD_FOLDER']+"/"+path_user
                    file_path = os.path.join(path, query['source_file'])
                    if search_File(path_user, query['source_file']):
                        result = compiler.create_table(f"user_{username}", query, file_path)
                        add_table_to_user(username, query['table_name'],query['columns'], query['source_file'])
                    else:
                        raise ManageExceptions("Ups! Archivo no encontrado")
                    
                elif query['action'] == 'CREATE INDEX':
                    path_user = str(generate_folder_path(username))
                    if query['method'] == 'GIN' and search_userTable(generate_folder_path(username), query['table_name']):
                        if verifify_exist_index(path_user,"table",query['table_name'],query['method']):
                            raise ManageExceptions("Ups! Indice ya generado para esta table")
                        compiler.create_index_psql(f"user_{username}", query)
                        add_index_to_table(username, query['table_name'], query['index_name'], query['method'], query['columns'], query['columns'])
                    
                    else:
                        source_file = ""
                        if query.get('table_name'):
                            source_file = search_userTable(generate_folder_path(username), query['table_name'])                            
                        else:
                            source_file = query['dataset']
                        
                        if search_File(path_user, source_file):
                            path = app.config['UPLOAD_FOLDER']+"/"+path_user
                            file_path = os.path.join(path, source_file)
                            print("Creando indice propio")
                            
                            print("Agregando a mongodb")
                            if query.get('table_name'):
                                if verifify_exist_index(path_user,"table",query['table_name'],query['method']):
                                    raise ManageExceptions("Ups! Indice ya generado para esta table")                                
                                compiler.create_index_own(file_path, query, path)
                                add_index_to_table(username, query['table_name'], query['index_name'], query['method'], query['columns'],query['language'])
                            else:
                                if verifify_exist_index(path_user,"dataset",query['dataset'],query['method']):
                                    raise ManageExceptions("Ups! Indice ya generado para este dataset")                                
                                compiler.create_index_own(file_path, query, path)
                                add_index_to_dataset(username, query['dataset'], query['index_name'], query['columns'],query['language'])
                        else:
                            raise ManageExceptions("Ups! Archivo no encontrado")
            return jsonify({"status": "ok", "result":result, "indexes":search_indexes(generate_folder_path(username)), "folder":search_userFolder(generate_folder_path(username))['folder']}), 200
        elif body['format'] == "text":
            if body['index'] == "postgres": 
                index = get_properties(body['indexes'],'gin', body['dataset'])['index']
                print("Indice recuperado: ",index)
                field_1 = "user_"+username
                field_2 =body['dataset']
                field_3 = body['limit']
                result = compiler.search_psql(field_1,textQuery,index,field_2,field_3)
            else:
                properties = get_properties(body['indexes'],'own', body['dataset'])
                columns_p = properties['columns']
                language_p = properties['language']
                source_file = search_userTable(generate_folder_path(username), body['dataset'])
                if source_file:
                    file_spimi = source_file
                else:
                    file_spimi = body['dataset']
                path_user = str(generate_folder_path(username))
                path = app.config['UPLOAD_FOLDER']+"/"+path_user
                file_path = os.path.join(path, file_spimi)
                #create_index_own(file_path, query, path)
                print(file_path,body['inputText'], columns_p,body['dataset'],body['limit'],language_p, path)
                result = compiler.search_spimi(file_path,body['inputText'], columns_p,body['dataset'],body['limit'],language_p, path)
            
        
        else:
            if body['index'] == "seq":
                result = compiler.search_knn_seq("", int(body['limit']))
                
            elif body['index'] == "rtree":
                #images = compiler.search_rtree(textQuery)
                print("")
            else:
                #images = compiler.search_hd(textQuery)
                print("")

        print({"status": "ok", "result":result, "images":images, "folder":search_userFolder(generate_folder_path(username))['folder']})
        return({"status": "ok", "result":result, "images":images, "folder":search_userFolder(generate_folder_path(username))['folder'], "indexes": search_indexes(generate_folder_path(username))}), 200
        
    except Exception as e:
        print("Error: ", e)
        return jsonify({"message": str(e)}), 500
    
if __name__ == "__main__":
    """
    Para correr el servidor Flask:
    . .venv/bin/activate
    flask --app main --debug run
    """
    app.run(debug=True, port=5000)
