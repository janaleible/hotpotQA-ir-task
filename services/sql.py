def create_table(name: str = 'retrievals') -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS  {name}
    (id INTEGER  AUTOINCREMENT, q_id TEXT, type TEXT, level TEXT, target_titles BLOB, result_int_ids BLOB)
    """


def insert_row(table: str = 'retrievals') -> str:
    return f"""
    INSERT INTO {table} (q_id, type, level, target_titles, result_int_ids) VALUES (?, ?, ?, ?, ?)
    """


def get_question_id(table: str = 'retrievals') -> str:
    return f"""
    SELECT q_id FROM {table} WHERE q_id = ?
    """


def get_retrieval_by_qid(table: str = 'retrievals') -> str:
    return f"""
    SELECT result_int_ids FROM {table} WHERE q_id = ?
    """


def get_question_by_qid(table: str = 'retrievals') -> str:
    return f"""
    SELECT q_id, target_titles, result_int_ids FROM {table} WHERE q_id = ?
    """


def get_count(table: str = 'retrievals') -> str:
    return f"""
    SELECT COUNT(id) FROM {table} 
    """


def get_reference(table: str = 'retrievals') -> str:
    return f"""
    SELECT q_id, target_titles FROM {table} 
    """

def get_retrievals(table: str = 'retrievals') -> str:
    return f"""
    SELECT q_id, result_int_ids FROM {table}
    """