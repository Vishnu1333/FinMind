import os
from sqlalchemy import create_engine, inspect
import psycopg2

db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/finmind")
engine = create_engine(db_url)
inspector = inspect(engine)

print("Tables:", inspector.get_table_names())
