from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os

load_dotenv()

username = os.getenv("username")
password = quote_plus(os.getenv("password"))
host = os.getenv("host")
database = os.getenv("database")

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}/{database}"
)