CREATE_TABLE_IF_NOT_EXISTS = """
    CREATE TABLE IF NOT EXISTS articles 
    (title TEXT PRIMARY KEY, article BLOB )
    """

INSERT_PREPROCESSED = """
        INSERT INTO articles (title, article)
        VALUES (?, ?)
    """
