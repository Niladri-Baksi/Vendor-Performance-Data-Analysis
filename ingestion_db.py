import pandas as pd
import os
from sqlalchemy import create_engine
import logging
import time
from urllib.parse import quote_plus
from dotenv import load_dotenv

logging.basicConfig(
    filename="logs/ingestion_db.log",
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'  #append 
    ) 

def ingest_db(df,table_name,engine):
    '''for ingesting dataframe into the database table'''
    df.to_sql(table_name, con=engine, if_exists='replace', index=False) #if_exists=append when continuous data insertion needed
    

load_dotenv()

username = os.getenv("DB_username")
password = quote_plus(os.getenv("DB_password"))
host = os.getenv("DB_host")
database = os.getenv("DB_database")

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}/{database}"
)
folder_path = 'C:\\Niladri\\empty\\coding\\projects\\Vendor Performance Data Analysis\\dataset\\data\\data'

def load_raw_data():
    '''load csv files from the dataset folder and ingest into the database'''
    start = time.time()
    for file in os.listdir(folder_path):
        # print(file)
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            logging.info(f"Ingesting {file} in dB")
            ingest_db(df, file[:-4], engine) #tablename=file and removing the .csv extension
            #df.to_sql(file[:-4], con=engine, if_exists='replace', index=False)
    end = time.time()
    total_time = (end - start)/60
    logging.info(f"All files ingested successfully in {total_time:.2f} minutes")

if __name__ == "__main__":
    load_raw_data()

#what this does is if its run directly then it calls the fn but if its imported as a module it skips the fn call and only defines the fn which can be called from the importing module.
#this is useful when we want to use the ingest_db fn in other modules without running the load_raw_data fn hence skipping the heavy processing of the files again and again.

