
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from engine import engine
import ingestion_db


logging.basicConfig(
    filename="logs/get_vendor_summary.log",
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'  #append 
    ) 

def get_vendor_sales_summary(conn):
    '''This function generates a vendor-wise sales and purchase summary, which is valuable for:
    Performance Optimization:
    The query involves heavy joins and aggregations on large datasets like sales and purchases.
    Storing the pre-aggregated results avoids repeated expensive computations.
    Helps in analyzing sales, purchases, and pricing for different vendors and brands.
    Future Benefits of Storing this data for faster Dashboarding & Reporting.
    Instead of running expensive queries each time, dashboards can fetch data quickly from vendor_sales_summary.'''
    vendor_sales_summary = pd.read_sql_query("""WITH FreightSummary AS (
        SELECT
            VendorNumber,
            SUM(Freight) AS TotalFreightCost
        FROM vendor_invoice
        GROUP BY VendorNumber
    ),
                                            
        PurchaseSummary AS (
            SELECT
                p.VendorNumber,
                p.VendorName,
                p.Brand,
                p.Description,
                p.PurchasePrice,
                pp.Volume,
                pp.Price AS ActualPrice,
                SUM(p.Quantity) AS TotalPurchaseQuantity,
                SUM(p.Dollars) AS TotalPurchaseDollars
            FROM purchases p
            JOIN purchase_prices pp
                ON p.Brand = pp.Brand
                AND p.VendorNumber = pp.VendorNumber
            WHERE p.PurchasePrice > 0
            GROUP BY
                p.VendorNumber,
                p.VendorName,
                p.Brand,
                p.Description,
                p.PurchasePrice,
                pp.Volume,
                pp.Price
        ),
                                            
        SalesSummary AS (
            SELECT
                s.VendorNo,
                s.Brand,
                SUM(s.SalesQuantity) AS TotalSalesQuantity,
                SUM(s.SalesDollars) AS TotalSalesDollars,
                SUM(s.SalesPrice) AS TotalSalesPrice,
                SUM(s.ExciseTax) AS TotalExciseTax
            FROM sales s
            GROUP BY s.VendorNo, s.Brand
        )
            
    SELECT
        ps.VendorNumber,
        ps.VendorName,
        ps.Brand,
        ps.Description,
        ps.PurchasePrice,
        ps.Volume,
        ps.ActualPrice,
        ps.TotalPurchaseQuantity,
        ps.TotalPurchaseDollars,
        ss.TotalSalesQuantity,
        ss.TotalSalesDollars,
        ss.TotalSalesPrice,
        ss.TotalExciseTax,
        fs.TotalFreightCost
    FROM PurchaseSummary ps
    LEFT JOIN SalesSummary ss
        ON ps.VendorNumber = ss.VendorNo
        AND ps.Brand = ss.Brand
    LEFT JOIN FreightSummary fs
        ON ps.VendorNumber = fs.VendorNumber
    ORDER BY ps.TotalPurchaseDollars DESC
    """, con=engine)

    return vendor_sales_summary


def clean_data(df):
    '''This function performs data cleaning operations on the vendor_sales_summary dataframe, including:
    1. Handling Missing Values: It fills any missing values in the dataframe with 0, ensuring that subsequent analyses are not affected by null entries.
    2. Standardizing Vendor Names: It removes leading and trailing spaces from the 'VendorName' column, which helps in maintaining consistency and avoiding issues during grouping or merging operations.'''
    df['Volume']=df['Volume'].astype(float)
    #df.dtypes
    df.fillna(0, inplace=True)
    df['VendorName']=df['VendorName'].str.strip()
    df ['VendorName'].unique()
    #df
    df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
    #df['GrossProfit'].min()
    df['ProfitMargin'] = (df['GrossProfit'] / df['TotalSalesDollars']) * 100
    df['StockTurnover'] = df['TotalSalesQuantity'] / df['TotalPurchaseQuantity']
    df['SalestoPurchaseRatio'] = df['TotalSalesDollars'] / df['TotalPurchaseDollars']

    return df


if __name__ == "__main__":
    logging.info('Creating Vendor Summary Table.....')

    summary_df = get_vendor_sales_summary(engine)
    logging.info(summary_df.head())

    logging.info('Cleaning Data.....')
    clean_df = clean_data(summary_df)
    logging.info(clean_df.head())

    logging.info('Ingesting data.....')
    # Replace infinities
    clean_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )
    ingestion_db.ingest_db(
        clean_df,
        'vendor_sales_summary',
        engine
    )

    logging.info('Completed')


#This script performs the following steps:
#1. Retrieves a vendor-wise sales and purchase summary from the database using a complex SQL query
#2. Cleans the retrieved data by handling missing values and standardizing vendor names
#3. Ingests the cleaned data back into the database for future use in dashboards and reporting, avoiding the need for expensive computations on large datasets each time.
    #The table vendor_sales_summary is deleted and created again and again in MySQL each time this script is run because of the if_exists='replace' parameter in the to_sql function. This ensures that we always have the most up-to-date summary data in the database, reflecting any changes in the underlying sales and purchase data. However, if you want to keep historical data, you can change it to if_exists='append' and add a timestamp column to track when each summary was generated or if_exists='fail' to avoid overwriting existing data and raise an error if the table already exists.

#The script is designed to be run directly, and it will execute the main block of code that generates the vendor summary, cleans the data, and ingests it into the database. If this script is imported as a module in another script, the main block will not execute, allowing you to use the get_vendor_sales_summary and clean_data functions without running the entire data processing workflow. This modular design promotes code reusability and separation of concerns.

#The notebook is used for exploratory data analysis (EDA) on the vendor_sales_summary table, where we can perform various analyses, visualizations, and insights generation based on the cleaned and ingested data.

#The script prepares the data and the notebook allows us to explore and analyze the data, which can be used for creating dashboards, reports, or further insights into vendor performance.

#The script reads from the raw table does ETL and writes to a new table vendor_sales_summary every time which is then read in the EDA notebook for analysis and visualization.

#Raw Tables(sales,purchases,...) -> EDA Script -> vendor_sales_summary table(ingested to MySQL) -> EDA Notebook -> Visualization + Insights + ML