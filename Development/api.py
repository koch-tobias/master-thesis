from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse

import pandas as pd

from pyxlsb import open_workbook as open_xlsb
from typing import Annotated, Union
from io import BytesIO
from loguru import logger

from model_predictions import predict_on_new_data


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
        benennung = row['Benennung (dt)']
        einheitsname = row['Einheitsname']
        ausfuehrung = row["Linke/Rechte Ausfuehrung"]

        result_dict[sachnummer] = [benennung, einheitsname]
        if (ausfuehrung == 'Linke Ausfuehrung') or (ausfuehrung == 'Rechte Ausfuehrung'):
            result_dict[sachnummer].append(ausfuehrung)

    return result_dict    

@app.post("/api/get_relevant_parts/")
async def post_relevant_parts(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(BytesIO(contents))

    df_predicted, einheitsname_not_found, ncars = predict_on_new_data(df)
    df_json = dataframe_to_dict(df_predicted)
    
    return df_json


@app.get("/get_relevant_parts/{file_path:path}")
async def get_relevant_parts(file_path: str):
    try:
        df = pd.read_excel(file_path, header=None, skiprows=1)
    except:
        raise UnicornException(name=f"Load Excel to dataframe failed!")

    df_predicted, einheitsname_not_found, ncars = predict_on_new_data(df)
    df_json = df_predicted.to_dict(orient='series') 

    return df_json