import pandas as pd

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse

from pyxlsb import open_workbook as open_xlsb
from io import BytesIO

from src.deployment.inference import Identifier
from src.utils import read_file

app = FastAPI() # Initialize api

# Class to through exceptions
class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

# Function to create costumized exceptions
@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=404,
        content={"message": f"{exc.name}"},
    )

# Function to convert a dataframe to a dictionary
def dataframe_to_dict(df, einheitsname_not_found):
    result_dict = {}
    for index, row in df.iterrows():
        sachnummer = row['Sachnummer']
        designation = row['Benennung (dt)']
        einheitsname = row['Einheitsname']
        zi = row['Zeichnungsindex']
        doku = row['Doku-Teil']
        alternative = row['Alternative']
        dok_format = row['Dok-Format']
        result_dict[sachnummer] = [designation, zi, doku, alternative, dok_format, einheitsname]
    
    result_dict["Fehlende Bauteile"] = einheitsname_not_found

    return result_dict    

# Post function that the user can use to sent data to the api and get back the list of all relevant car parts in json format
@app.post("/api/get_relevant_parts/")
async def post_relevant_parts(file: UploadFile = File(...)):
    if file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
        try:
            contents = await file.read()
            df, ncar = read_file(BytesIO(contents), raw=True)         
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Error reading file. Make sure it is a valid Excel file."})
      
        try:
            df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar = Identifier.classification_on_new_data(df)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Error in identifying the relevant components."})
        
        try:
            df_json = dataframe_to_dict(df_relevant_parts, einheitsname_not_found)
            return JSONResponse(status_code=200, content=df_json)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Error converting the file to json format."})
        
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid file extension. Only Excel files (.xlsx or .xls) are accepted."})

# Get function to test if the api is running and if the connection is successfull 
@app.get("/")
async def root():
    return {"message": "Hello User, your connection to the API is successfull!"}