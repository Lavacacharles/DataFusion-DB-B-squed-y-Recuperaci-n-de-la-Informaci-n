import re
from typing import List

def extract_type(s: str) -> int:
    # Simulación del tipo de archivo
    print(f"Extracting type for {s}")
    if "hash" in s.lower():
        return 0
    elif "avl" in s.lower():
        return 1
    elif "sequential" in s.lower():
        return 2
    return -1

def extract_numbers_between(text: str) -> List[str]:
    pattern = r"(\d+)\s+and\s+(\d+)"
    match = re.search(pattern, text)
    if match:
        return [match.group(1), match.group(2)]
    return []

def separate_id_data(text: str) -> List[str]:
    parts = text.split(',', 1)
    return parts if len(parts) == 2 else []

class SQLCompiler:
    def __init__(self):
        pass

    def _trim(self, text: str) -> str:
        return text.strip()

    def _split_string(self, text: str, delimiter: str) -> List[str]:
        return [self._trim(token) for token in text.split(delimiter)]

    def _validate_create_table(self, statement: str) -> List[str]:
        pattern = r'create\s+table\s+(\w+)\s+from\s+file\s+"([^"]+)"\s+using\s+index\s+(\w+)\("(\w+)"\)'
        match = re.search(pattern, statement, re.IGNORECASE)
        
        if not match:
            return ["Invalid CREATE TABLE syntax"]
        
        table_name, file_path, index_type, index_column = match.groups()
        print(f"CREATE TABLE operation detected")
        print(f"Table name: {table_name}")
        print(f"File path: {file_path}")
        print(f"Index type: {index_type}")
        print(f"Index column: {index_column}")
        
        if index_type == "hash":
            print("Creating hash table...")
        elif index_type == "avl":
            print("Creating AVL table...")
        elif index_type == "sequential":
            print("Creating sequential file...")
            
        return ["CREATE TABLE operation detected", table_name, "Table created successfully"]

    def _validate_select(self, statement: str) -> List[str]:
        pattern = r'select\s+from\s+(\w+)\s+where\s+(\w+)\s*(=|between)\s*(.*)'
        match = re.search(pattern, statement, re.IGNORECASE)
        
        if not match:
            return ["Invalid SELECT syntax"]
            
        table_name, column, operator, value = match.groups()
        
        if operator == "between":
            print("SEARCH RANGE operation detected")
            table_type = extract_type(table_name)
            print(f"Table type: {table_type}")
            range_values = extract_numbers_between(value)
            print(f"Range search: {range_values[0]} to {range_values[1]}")
            return ["SEARCH RANGE operation detected", f"Found records in range {range_values[0]} to {range_values[1]}"]
        else:
            print("SEARCH operation detected")
            table_type = extract_type(table_name)
            print(f"Table type: {table_type}")
            print(f"Searching for key: {value}")
            return ["SEARCH operation detected", f"Found record with key {value}"]

    def _validate_insert(self, statement: str) -> List[str]:
        pattern = r'insert\s+into\s+(\w+)\s+values\s*\((.*)\)'
        match = re.search(pattern, statement, re.IGNORECASE)
        
        if not match:
            return ["Invalid INSERT syntax"]
            
        table_name, values = match.groups()
        print("INSERT operation detected")
        table_type = extract_type(table_name)
        print(f"Table type: {table_type}")
        print(f"Inserting values: {values}")
        return ["INSERT operation detected", f"Inserted values: {values}"]

    def _validate_delete(self, statement: str) -> List[str]:
        pattern = r'delete\s+from\s+(\w+)\s+where\s+(\w+)\s*=\s*(.*)'
        match = re.search(pattern, statement, re.IGNORECASE)
        
        if not match:
            return ["Invalid DELETE syntax"]
            
        table_name, column, value = match.groups()
        print("REMOVE operation detected")
        table_type = extract_type(table_name)
        print(f"Table type: {table_type}")
        print(f"Deleting record with key: {value}")
        return ["REMOVE operation detected", f"Deleted record with key {value}"]

    def process_query(self, query: str) -> str:
        statements = self._split_string(query, ';')
        results = []
        
        for statement in statements:
            if not statement:
                continue
                
            try:
                command = statement.strip().lower().split()[0]
                if command == "create":
                    results.append(self._validate_create_table(statement))
                elif command == "select":
                    results.append(self._validate_select(statement))
                elif command == "insert":
                    results.append(self._validate_insert(statement))
                elif command == "delete":
                    results.append(self._validate_delete(statement))
                else:
                    print(f"Unknown command: {command}")
            except Exception as e:
                print(f"Error processing statement: {statement}")
                print(f"Error: {str(e)}")
                
        return 'name,age,city,occupation,salary,department\nJohn,30,New York,Engineer,70000,R&D\nJane,25,Boston,Doctor,85000,Health\nDoe,22,San Francisco,Artist,50000,Art\nAlice,29,Chicago,Teacher,60000,Education\nBob,34,Seattle,Nurse,55000,Health\nCharlie,28,Austin,Architect,75000,Construction\nDiana,40,Denver,Scientist,95000,Research\nEve,27,Miami,Lawyer,67000,Law\nFrank,26,Orlando,Chef,52000,Hospitality\nGrace,32,Dallas,Pilot,88000,Aviation\nJohn,30,New York,Engineer,70000,R&D\nJane,25,Boston,Doctor,85000,Health\nDoe,22,San Francisco,Artist,50000,Art\nAlice,29,Chicago,Teacher,60000,Education\nBob,34,Seattle,Nurse,55000,Health\nCharlie,28,Austin,Architect,75000,Construction\nDiana,40,Denver,Scientist,95000,Research\nEve,27,Miami,Lawyer,67000,Law\nFrank,26,Orlando,Chef,52000,Hospitality\nGrace,32,Dallas,Pilot,88000,Aviation\nJohn,30,New York,Engineer,70000,R&D\nJane,25,Boston,Doctor,85000,Health\nDoe,22,San Francisco,Artist,50000,Art\nAlice,29,Chicago,Teacher,60000,Education\nBob,34,Seattle,Nurse,55000,Health\nCharlie,28,Austin,Architect,75000,Construction\nDiana,40,Denver,Scientist,95000,Research\nEve,27,Miami,Lawyer,67000,Law\nFrank,26,Orlando,Chef,52000,Hospitality\nGrace,32,Dallas,Pilot,88000,Aviation\nJohn,30,New York,Engineer,70000,R&D\nJane,25,Boston,Doctor,85000,Health\nDoe,22,San Francisco,Artist,50000,Art\nAlice,29,Chicago,Teacher,60000,Education\nBob,34,Seattle,Nurse,55000,Health\nCharlie,28,Austin,Architect,75000,Construction\nDiana,40,Denver,Scientist,95000,Research\nEve,27,Miami,Lawyer,67000,Law\nFrank,26,Orlando,Chef,52000,Hospitality\nGrace,32,Dallas,Pilot,88000,Aviation\nJohn,30,New York,Engineer,70000,R&D\nJane,25,Boston,Doctor,85000,Health\nDoe,22,San Francisco,Artist,50000,Art\nAlice,29,Chicago,Teacher,60000,Education\nBob,34,Seattle,Nurse,55000,Health\nCharlie,28,Austin,Architect,75000,Construction\nDiana,40,Denver,Scientist,95000,Research\nEve,27,Miami,Lawyer,67000,Law\nFrank,26,Orlando,Chef,52000,Hospitality\nGrace,32,Dallas,Pilot,88000,Aviation';
        # return results

def main():
    compiler = SQLCompiler()
    
    # Test queries
    queries = [
        'create table customer from file "../datos_small.csv" using index hash("Codigo");',
        'create table customeravl from file "../datos_small.csv" using index avl("Codigo");',
        'select from Customer where Codigo = 1;',
        'select from CustomerAVL where Codigo = 1;',
        'select from Customer where Codigo between 22 and 32;',
        'select from CustomerAVL where Codigo between 22 and 32;',
        'insert into Customer values (3,John,Doe,5);',
        'insert into CustomerAVL values (3,John,Doe,5);',
        'select from Customer where Codigo = 3;',
        'select from CustomerAVL where Codigo = 3;',
        'delete from Customer where Codigo = 3;',
        'delete from CustomerAVL where Codigo = 3;'
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        results = compiler.process_query(query)
        print("Results:", results)
        print("-" * 50)

if __name__ == "__main__":
    main()