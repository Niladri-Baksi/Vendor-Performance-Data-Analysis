from sqlalchemy import create_engine
from urllib.parse import quote_plus

username = "root"
password = quote_plus("niladri-mysql@27")
host = "localhost"
database = "vendor_analysis"

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}/{database}"
)