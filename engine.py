from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os

load_dotenv()

username = os.getenv("DB_username")
password = quote_plus(os.getenv("DB_password"))
host = os.getenv("DB_host")
database = os.getenv("DB_database")

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}/{database}"
)