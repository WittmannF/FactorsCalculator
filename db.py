import gspread
import pandas as pd
import os
from google.cloud import bigquery
import google.auth
import unittest
import inspect
import pandas as pd


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='/Users/wittmann/.secrets/quickstart-1531738244681-a8d758fc4e43.json'

SCOPES=[
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/bigquery",
    ]

class BigQuery:
    def __init__(self):
        self.client=self.get_new_client()

    def query(self, q, output_type='dataframe'):
        """
        >>> q = "SELECT * FROM `dashboards-data-293720.test.test_load_table_from_dataframe` LIMIT 2"
        >>> bq.query(q)
           age workclass education_level  ...  hours_per_week native_country income
        0   49   Private             9th  ...            16.0        Jamaica  <=50K
        1   49   Private             9th  ...            16.0        Jamaica  <=50K
        <BLANKLINE>
        [2 rows x 14 columns]
        >>>
        """
        query_job = self.client.query(q)
        results = query_job.result()
        if output_type=='dataframe':
            results=results.to_dataframe()
        return results

    def get_new_client(self):
        credentials, project = google.auth.default(scopes=SCOPES)
        return bigquery.Client(credentials=credentials, project=project)

    def load_table_from_csv(self, file_path, table_id, write_disposition="WRITE_APPEND"):
        """
        table_id:
            Ex "dashboards-data-293720.test.test_load_table_from_dataframe"
        write_disposition:
            WRITE_EMPTY --> Writes the data only if the table is empty.
            WRITE_APPEND --> (Default) Appends the data to the end of the table.
            WRITE_TRUNCATE --> Erases all existing data in a table before writing the new data.
        """
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV, 
            skip_leading_rows=1, 
            autodetect=True,
            allow_quoted_newlines=True,
            write_disposition=write_disposition
        )
        with open(file_path, "rb") as source_file:
            job = self.client.load_table_from_file(source_file, table_id, job_config=job_config)

        print(job.result())

    def load_table_from_dataframe(self, df, table_id, write_disposition="WRITE_APPEND"):
        """
        table_id:
            Ex "dashboards-data-293720.test.test_load_table_from_dataframe"
        write_disposition:
            WRITE_EMPTY --> Writes the data only if the table is empty.
            WRITE_APPEND --> (Default) Appends the data to the end of the table.
            WRITE_TRUNCATE --> Erases all existing data in a table before writing the new data.
        """
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition
        )
        job = self.client.load_table_from_dataframe(df, 
            table_id, 
            job_config=job_config)
        print(job.result())

class _TestBigQuery(unittest.TestCase):
 
    def test_load_table_from_dataframe(self):
        func_name=inspect.currentframe().f_code.co_name
        print('Testing', func_name)
        table_id = "dashboards-data-293720.test.test_append"
        file_path='data_for_test.csv'
        df=pd.read_csv(file_path)
        df.columns = [c.replace('-', '_') for c in df.columns]

        bq = BigQuery()
        bq.load_table_from_dataframe(df, 
            table_id, 
            write_disposition="WRITE_TRUNCATE")


    def test_load_table_from_csv(self):
        func_name=inspect.currentframe().f_code.co_name
        print('Testing', func_name)
        table_id = "dashboards-data-293720.test.test_csv"
        file_path='data_for_test.csv'

        bq = BigQuery()
        bq.load_table_from_csv(file_path, 
            table_id, 
            write_disposition="WRITE_TRUNCATE")



    def test_query(self):
        print('Testing query job')
        bq = BigQuery()
        q = "SELECT * FROM `dashboards-data-293720.test.test_append`"
        df=bq.query(q)
        print(df)

class Spreadsheet:
    def __init__(self, spreadsheet_name=None, worksheet=None):
        self.gc = gspread.oauth()
        if spreadsheet_name is not None:
            self.sh = self.gc.open(spreadsheet_name)
        else:
            self.sh = None

        if worksheet is not None:
            self.ws = self.sh.worksheet(worksheet)
        else:
            self.ws = None

    def open_spreadsheet(self, spreadsheet_name):
        self.sh = self.gc.open(spreadsheet_name)
    
    def open_worksheet(self, worksheet):
        self.ws = self.gc.worksheet(worksheet)
    
    def get_data(self, range_or_val, worksheet=None, return_df=True):
        if worksheet is not None:
            data = self.sh.worksheet(worksheet).get(range_or_val)
        else:
            data = self.ws.get(range_or_val)
        if return_df:
            try:
                return pd.DataFrame(data[1:], columns=data[0])
            except Exception as e:
                print('Returing DF failed:')
                print(e)
                return data
        else:
            return data
    
    def write_to_worksheet(self, range, values, worksheet=None, value_input_option='USER_ENTERED'):
        if worksheet is not None:
            self.ws.update(
                range, 
                values, 
                value_input_option=value_input_option
            )
        else:
            self.sh.worksheet(
                worksheet
            ).update(
                range, 
                values, 
                value_input_option=value_input_option
            )
    
    def write_dataframe(self, dataframe):
        self.ws.update([dataframe.columns.values.tolist()] + dataframe.values.tolist())
