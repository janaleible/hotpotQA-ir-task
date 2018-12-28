import sqlite3
import main_constants as ct

if __name__ == '__main__':
    db = sqlite3.connect(ct.TEST_FEATURES_DB)
    cursor = db.cursor()
    cursor.execute("CREATE INDEX doc_title_text ON features (doc_title, document_text)")
    db.commit()
    cursor.close()
    db.close()

