from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse

import pandas as pd

from pyxlsb import open_workbook as open_xlsb
from typing import Annotated, Union
from io import BytesIO
from loguru import logger

from models.predict import predict_on_new_data


class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

app = FastAPI()

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=404,
        content={"message": f"{exc.name}"},
    )

def dataframe_to_dict(df):
    result_dict = {}
    for index, row in df.iterrows():
        sachnummer = row['Sachnummer']
        einheitsname = row['Einheitsname']
        '''
        benennung = row['Benennung (dt)']
        ausfuehrung = row["Linke/Rechte Ausfuehrung"]
        result_dict[sachnummer] = [benennung, einheitsname]
        if (ausfuehrung == 'Linke Ausfuehrung') or (ausfuehrung == 'Rechte Ausfuehrung'):
            result_dict[sachnummer].append(ausfuehrung)
        '''
        result_dict[sachnummer] = einheitsname

    return result_dict    

@app.post("/api/get_relevant_parts/")
async def post_relevant_parts(file: UploadFile = File(...)):
    if file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
        try:
            contents = await file.read()
            df = pd.read_excel(BytesIO(contents))
            df.columns = df.iloc[0]
            df = df.iloc[1:]
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Fehler beim Lesen der Datei. Stellen Sie sicher, dass es sich um eine gültige Excel-Datei handelt."})
      
        try:
            df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar = predict_on_new_data(df, use_api=True)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Fehler bei der Vorhersage."})
        
        try:
            df_json = dataframe_to_dict(df_relevant_parts)
            return JSONResponse(status_code=200, content=df_json)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Fehler beim umwandeln der Datei."})
        
    else:
        return JSONResponse(status_code=400, content={"error": "Ungültige Dateierweiterung. Es werden nur Excel-Dateien (.xlsx oder .xls) akzeptiert."})