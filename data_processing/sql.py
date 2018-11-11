CREATE_TABLE_IF_NOT_EXISTS = """
    CREATE TABLE IF NOT EXISTS articles 
    (title TEXT PRIMARY KEY, article BLOB )
    """

def insert_string(title, article):
    return f"""
        INSERT INTO articles
        VALUES {title}, {article}
    """