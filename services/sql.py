def create_table(name: str = 'retrievals') -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS  {name}
    (id TEXT PRIMARY KEY, type TEXT, level TEXT, target_titles BLOB, result_int_ids BLOB)
    """


def insert_row(table: str = 'retrievals') -> str:
    return f"""
    INSERT INTO {table} VALUES (?, ?, ?, ?, ?)
    """


def get_question_id(table: str = 'retrievals') -> str:
    return f"""
    SELECT id FROM {table} WHERE id = ?
    """


def get_retrieval(table: str = 'retrievals') -> str:
    return f"""
    SELECT result_int_ids FROM {table} WHERE id = ?
    """


def get_count(table: str = 'retrievals') -> str:
    return f"""
    SELECT COUNT(id) FROM {table} 
    """


def get_reference(table: str = 'retrievals') -> str:
    return f"""
    SELECT id, target_titles FROM {table} 
    """

def get_retrievals(table: str = 'retrievals') -> str:
    return f"""
    SELECT id, result_int_ids FROM {table}
    """