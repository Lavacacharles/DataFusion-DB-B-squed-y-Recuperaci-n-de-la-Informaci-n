import ply.lex as lex
import ply.yacc as yacc
import json

# Definir las palabras reservadas
reserved = {
    'use': 'USE',
    'create': 'CREATE',
    'table': 'TABLE',
    'index': 'INDEX',
    'on': 'ON',
    'using': 'USING',
    'and': 'AND',
    'from': 'FROM',
    'dataset': 'DATASET',  # Asegurarse de que esté incluido
    # 'language', 'own' no están incluidos para tratarlos como IDENTIFIER
}

# Lista de tokens
tokens = [
    'IDENTIFIER',
    'STRING',
    'LPAREN',
    'RPAREN',
    'COMMA',
    'NUMBER',
    'SEMICOLON',
] + list(set(reserved.values()))

# Expresiones regulares para tokens simples
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COMMA = r','
t_SEMICOLON = r';'
t_STRING = r'"[^"]*"'
t_NUMBER = r'\d+'

# Ignorar espacios y tabulaciones
t_ignore = ' \t\r\n'

# Token IDENTIFIER y verificación de palabras reservadas (case-insensitive)
def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t_lower = t.value.lower()
    t.type = reserved.get(t_lower, 'IDENTIFIER')  # Verifica si es una palabra reservada
    t.value = t.value  # Conserva el valor original
    return t

# Manejo de errores
def t_error(t):
    print(f"Illegal character {repr(t.value[0])} at line {t.lineno}")
    t.lexer.skip(1)

# Crear el lexer
lexer = lex.lex()

# Gramática
def p_statements(p):
    """statements : statements statement SEMICOLON
                  | statement SEMICOLON"""
    if len(p) == 4:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_statement(p):
    """statement : use_statement
                 | create_table_statement
                 | create_index_statement
                 | create_dataset_statement"""
    p[0] = p[1]

def p_use_statement(p):
    "use_statement : USE STRING"
    p[0] = {"action": "USE", "database": p[2].strip('"')}

def p_create_dataset_statement(p):
    "create_dataset_statement : CREATE DATASET FROM STRING"
    p[0] = {
        "action": "CREATE DATASET",
        "source_file": p[4].strip('"')
    }

def p_create_table_statement(p):
    "create_table_statement : CREATE TABLE IDENTIFIER LPAREN columns RPAREN FROM STRING"
    p[0] = {
        "action": "CREATE TABLE",
        "table_name": p[3],
        "columns": p[5],
        "source_file": p[8].strip('"')
    }

def p_create_index_statement(p):
    """create_index_statement : CREATE INDEX STRING ON IDENTIFIER USING IDENTIFIER LPAREN IDENTIFIER RPAREN AND IDENTIFIER LPAREN STRING RPAREN
                              | CREATE INDEX STRING ON IDENTIFIER USING IDENTIFIER LPAREN columns RPAREN AND IDENTIFIER LPAREN STRING RPAREN
                              | CREATE INDEX STRING ON DATASET STRING USING IDENTIFIER LPAREN columns RPAREN AND IDENTIFIER LPAREN STRING RPAREN
                              | CREATE INDEX STRING ON DATASET STRING USING IDENTIFIER LPAREN IDENTIFIER RPAREN AND IDENTIFIER LPAREN STRING RPAREN"""
    if p[4].lower() == 'on' and p[5].lower() == 'dataset':
        # Manejar la estructura con DATASET
        if p[12].lower() != 'and':
            print(f"Syntax error: Expected 'AND', got {p[12]}")
        if p[13].lower() != 'language':
            print(f"Syntax error: Expected 'LANGUAGE', got {p[13]}")
        p[0] = {
            "action": "CREATE INDEX",
            "index_name": p[3].strip('"'),
            "dataset": p[6].strip('"'),
            "method": p[8],
            "columns": p[10],
            "language": p[15].strip('"')
        }
    else:
        # Manejar la estructura anterior
        if p[12].lower() != 'language':
            print(f"Syntax error: Expected 'LANGUAGE', got {p[12]}")
        p[0] = {
            "action": "CREATE INDEX",
            "index_name": p[3].strip('"'),
            "table_name": p[5],
            "method": p[7],
            "columns": p[9],
            "language": p[14].strip('"')
        }

def p_columns(p):
    """columns : columns COMMA column
               | column"""
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_column(p):
    """column : IDENTIFIER IDENTIFIER LPAREN NUMBER RPAREN
              | IDENTIFIER IDENTIFIER
              | IDENTIFIER"""
    if len(p) == 6:
        p[0] = {"name": p[1], "type": f"{p[2]}({p[4]})"}
    elif len(p) == 3:
        p[0] = {"name": p[1], "type": p[2]}
    else:
        p[0] = p[1]  # Para el caso de columnas sin tipo

def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}' (token type: {p.type})")
    else:
        print("Syntax error at EOF")

# Crear el parser
parser = yacc.yacc()

# Función para procesar consultas y devolver como JSON
class ManageExceptions(Exception):
    """Error de Syntax """
    pass

def process_query(query):
    try:
        parsed = parser.parse(query)  # Ajusta `parser.parse` según tu implementación
        if not parsed:
            raise ManageExceptions("Ups! Query Syntax Error")
        return json.dumps(parsed, indent=4)
    except ManageExceptions as e:
        raise e
    except Exception as e:
        raise ManageExceptions(f"Error al procesar la consulta: {e}")

# Ejemplo de entrada
query = '''
CREATE DATASET FROM "file_path.csv";
CREATE TABLE songs (
    track_id varchar(22),
    lyrics varchar(27698),
    track_name varchar(123),
    track_artist varchar(51),
    track_album_name varchar(151),
    playlist_name varchar(120),
    playlist_genre varchar(5),
    playlist_subgenre varchar(25),
    language varchar(2)
) FROM "songs.csv";
CREATE INDEX "content_idx" ON songs USING GIN(lyrics) AND LANGUAGE("spanish"); INDICE PARA POSTGRESS
CREATE INDEX "content_idx" ON songs USING GIN(lyrics, name) AND LANGUAGE("spanish"); 
CREATE INDEX "content_idx" ON songs USING MATCHCORE(lyrics) AND LANGUAGE("spanish"); INDICE EN DOC ASOCIADO A TABLA
CREATE INDEX "content_idx" ON songs USING MATCHCORE(lyrics, name) AND LANGUAGE("spanish"); 
CREATE INDEX "content_idx" ON DATASET "songs.csv" USING MATCHCORE(lyrics) AND LANGUAGE("spanish"); INDICE EN DOCUMENTO DIRECTAMENTE
CREATE INDEX "content_idx" ON DATASET "songs.csv" USING MATCHCORE(lyrics, name) AND LANGUAGE("spanish");
'''

# Procesar la entrada y obtener el JSON
#query_json = process_query(query)
#print(query_json)
