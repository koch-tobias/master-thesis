# %%
import requests
import time

url = "http://127.0.0.1:8000/api/get_relevant_parts/" 
start = time.time()
file_path = "C:/Users/q617269/Desktop/Masterarbeit_Tobias/repos/master-thesis/data/raw/G20_prismaexport-20230621-143916.xls"
files = {"file": open(file_path, "rb")}
headers = {"accept": "application/json"}

proxies = {
  "http": None,
  "https": None}

response = requests.post(url, files=files, headers=headers, proxies=proxies)
stop = time.time()
training_time = stop - start
print(response.content)

# %%
'''
Sub SendRequestToAPI()
    Dim url As String
    Dim file_path As String
    Dim http_request As Object
    
    url = "http://localhost:8000/api/get_relevant_parts/"
    file_path = "path/to/excel/file.xlsx"
    
    ' Erstelle eine Instanz der WinHTTPRequest
    Set http_request = CreateObject("WinHTTP.WinHTTPRequest.5.1")
    
    ' Öffne die Datei als binären Datenstrom
    Open file_path For Binary Access Read As #1
    file_contents = Space$(LOF(1))
    Get #1, , file_contents
    Close #1
    
    ' Sende die Anfrage an die API
    http_request.Open "POST", url, False
    http_request.setRequestHeader "Content-Type", "application/vnd.ms-excel"
    http_request.send file_contents
    
    ' Überprüfe die Antwort der API
    If http_request.Status = 200 Then
        ' Erfolgreiche Antwort erhalten
        MsgBox http_request.responseText
    Else
        ' Fehler bei der Anfrage
        MsgBox "Fehler: " & http_request.Status & " - " & http_request.statusText
    End If
End Sub
'''