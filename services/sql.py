from typing import Tuple, List

create_candidate_table = """CREATE TABLE IF NOT EXISTS candidates
    (
        id INTEGER PRIMARY KEY AUTOINCREMENT, question_id TEXT, type TEXT, level TEXT,
        doc_iid TEXT, doc_wid TEXT, doc_title TEXT, 
        question_text TEXT, doc_text TEXT, question_tokens TEXT, doc_tokens TEXT,
        tfidf TEXT, relevance TEXT
    )"""

insert_candidate = """INSERT INTO candidates 
    (question_id, type, level, doc_iid, doc_wid, doc_title, 
    question_text, doc_text, question_tokens, doc_tokens, 
    tfidf, relevance)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""

count_question_rows = """SELECT question_id, COUNT(*) FROM candidates GROUP BY question_id"""

fetch_candidate_batch = """SELECT * FROM candidates WHERE id >= ? AND id <= ?"""
fetch_candidate_by_id = """SELECT * FROM candidates WHERE id = ?"""


def create_features_table(columns: List[str]):
    c = ", ".join(col + " TEXT" for col in columns)
    return f'CREATE TABLE IF NOT EXISTS features (id INTEGER PRIMARY KEY AUTOINCREMENT, {c})'


def insert_features(columns: List[str]):
    return f'INSERT INTO features ({", ".join(col for col in columns) }) VALUES ({", ".join(["?"] * (len(columns)))})'


def create_table(name: str = 'retrievals') -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS  {name}
    (id INTEGER PRIMARY KEY AUTOINCREMENT, q_id TEXT, type TEXT, level TEXT, target_titles BLOB, result_int_ids BLOB)
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
