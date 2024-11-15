import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class CreateTableParams:
    table_name: str
    file_path: str
    columns: List[str] = None

@dataclass
class CreateIndexParams:
    index_name: str
    table_name: str
    column_name: str
    language: str

@dataclass
class SelectParams:
    columns: List[str]
    table_name: str
    search_column: str
    search_phrase: str
    limit: int

class SQLParameterExtractor:
    def __init__(self):
        self.last_error = None
        
    def _clean_string(self, text: str) -> str:
        """Limpia el texto de comillas extras y espacios"""
        return text.strip().strip('"').strip("'")

    def extract_create_table_params(self, query: str) -> Optional[CreateTableParams]:
        """
        Extrae parámetros de CREATE TABLE
        Ejemplo: CREATE TABLE employees FROM FILE "data.csv";
        """
        pattern = r'create\s+table\s+(\w+)\s+from\s+file\s+["\']([^"\']+)["\']'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if not match:
            self.last_error = "Invalid CREATE TABLE syntax"
            return None
            
        table_name, file_path = match.groups()
        return CreateTableParams(
            table_name=self._clean_string(table_name),
            file_path=self._clean_string(file_path)
        )

    def extract_create_index_params(self, query: str) -> Optional[CreateIndexParams]:
        """
        Extrae parámetros de CREATE INDEX
        Ejemplo: CREATE INDEX 'desc_index' ON 'employees' USING GIN('description') AND LANGUAGE('spanish');
        """
        pattern = r"create\s+index\s+['\"]([^'\"]+)['\"]\s+on\s+['\"]([^'\"]+)['\"]\s+using\s+gin\(['\"]([^'\"]+)['\"]\)\s+and\s+language\(['\"]([^'\"]+)['\"]\)"
        match = re.search(pattern, query, re.IGNORECASE)
        
        if not match:
            self.last_error = "Invalid CREATE INDEX syntax"
            return None
            
        index_name, table_name, column_name, language = match.groups()
        return CreateIndexParams(
            index_name=self._clean_string(index_name),
            table_name=self._clean_string(table_name),
            column_name=self._clean_string(column_name),
            language=self._clean_string(language)
        )

    def extract_select_params(self, query: str) -> Optional[SelectParams]:
        """
        Extrae parámetros de SELECT
        Ejemplo: SELECT name, description FROM 'employees' WHERE 'description' @@ 'engineer & software' LIMIT 10;
        """
        pattern = r"select\s+([^']+?)\s+from\s+['\"]([^'\"]+)['\"]\s+where\s+['\"]([^'\"]+)['\"]\s+@@\s+['\"]([^'\"]+)['\"]\s+limit\s+(\d+)"
        match = re.search(pattern, query, re.IGNORECASE)
        
        if not match:
            self.last_error = "Invalid SELECT syntax"
            return None
            
        columns_str, table_name, search_column, phrase, limit = match.groups()
        
        # Procesar columnas
        columns = [col.strip() for col in columns_str.split(',')]
        
        return SelectParams(
            columns=columns,
            table_name=self._clean_string(table_name),
            search_column=self._clean_string(search_column),
            search_phrase=self._clean_string(phrase),
            limit=int(limit)
        )

    def process_query(self, query: str) -> Tuple[str, Optional[object]]:
        """
        Procesa una query y retorna el tipo de operación y sus parámetros
        """
        query = query.strip()
        first_word = query.split()[0].lower()
        
        if first_word == "create":
            if "index" in query.lower():
                params = self.extract_create_index_params(query)
                return "CREATE_INDEX", params
            else:
                params = self.extract_create_table_params(query)
                return "CREATE_TABLE", params
        elif first_word == "select":
            params = self.extract_select_params(query)
            return "SELECT", params
        else:
            self.last_error = f"Unknown command: {first_word}"
            return "UNKNOWN", None

def main():
    # Ejemplos de uso
    extractor = SQLParameterExtractor()
    
    test_queries = [
        'CREATE TABLE employees FROM FILE "data/employees.csv";',
        'CREATE INDEX "content_idx" ON "employees" USING GIN("description") AND LANGUAGE("spanish");',
        'SELECT name, position, department FROM "employees" WHERE "description" @@ "software & engineer" LIMIT 5;'
    ]
    
    for query in test_queries:
        print("\nProcessing query:", query)
        operation, params = extractor.process_query(query)
        
        print(f"Operation: {operation}")
        if params:
            print("Parameters extracted:")
            for key, value in params.__dict__.items():
                print(f"  {key}: {value}")
        else:
            print(f"Error: {extractor.last_error}")
        print("-" * 50)

if __name__ == "__main__":
    main()