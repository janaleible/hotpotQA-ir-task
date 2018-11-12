CREATE_TABLE_IF_NOT_EXISTS = """
    CREATE TABLE IF NOT EXISTS articles 
    (id INTEGER PRIMARY KEY, title TEXT, doc_string TEXT, doc_blob BLOB, tokens BLOB)
    """

INSERT_EXTRACTED_DOC = """
        INSERT INTO articles (id, title, doc_string, doc_blob, tokens)
        VALUES (?, ?, ?, ?, ?)
    """

RETRIEVE_ALL = """
    SELECT title, tokens FROM articles
"""

COUNT_ALL = """
    SELECT COUNT(*) FROM articles
"""