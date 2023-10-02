from fastapi import FastAPI, File, UploadFile
from Ultils import ReturnInfoLP
from Ultils import check_type_image
import os
import time
import asyncio
from typing import List
import uvicorn
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp
from zipfile import ZipFile
from aiofiles import open as async_open
start_time = time.time()
class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_upload_size: int) -> None:
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == 'POST':
            if 'content-length' not in request.headers:
                return Response(status_code=status.HTTP_411_LENGTH_REQUIRED)
            content_length = int(request.headers['content-length'])
            if content_length > self.max_upload_size:
                return Response(status_code=status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE)
        return await call_next(request)
app = FastAPI()
app.add_middleware(LimitUploadSize, max_upload_size=10 * 1024 * 1024)  # giới hạn kích thước tải lên=10MB
@app.get("/")
def read_root():
    return {"Hello": "Demo by Nguyen Dai"}
@app.post("/LicencePlate/UploadingSingleFile")
async def UploadingSingleFile(file: UploadFile = File(...)):
    try:
        pathSave = os.path.join(os.getcwd(),'anhtoancanh')
        os.makedirs(pathSave,exist_ok=True)
        async with async_open(os.path.join(pathSave,file.filename),'wb') as f:
            await f.write(await file.read()) 
        obj = await asyncio.to_thread(ReturnInfoLP,os.path.join(pathSave,file.filename))
        if (obj.errorCode ==0):
            return {"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,
            "data": [{"textPlate": obj.textPlate, "accPlate": obj.accPlate, "imagePlate": obj.imagePlate}]}
        else:
            return {"errorCode": obj.errorCode, "message": obj.errorMessage, "data": []}
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        await file.close()
@app.post("/LicencePlate/UploadingMultipleFiles")
async def UploadingMultipleFiles(files: List[UploadFile]):
    pathSave = os.path.join(os.getcwd(), 'anhtoancanh')
    os.makedirs(pathSave, exist_ok=True)
    dict_values = {}
    for file in files:
        async with async_open(os.path.join(pathSave, file.filename), 'wb') as f:
            await f.write(await file.read())
        obj = await asyncio.to_thread(ReturnInfoLP, os.path.join(pathSave, file.filename))
        if (obj.errorCode ==0):
            dict_values.update({file.filename:{"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,
            "data": [{"textPlate": obj.textPlate, "accPlate": obj.accPlate, "imagePlate": 'anhbienso/'+obj.imagePlate}]}})
        else:
            dict_values.update({file.filename:{"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,"data": []}})
        file.file.close()
    return dict_values

@app.post("/LicencePlate/UploadingZipFile")
async def UploadingZipFile(file: UploadFile = File(...)):
    dict_values = {}
    pathSave = os.path.join(os.getcwd(), 'anhtoancanh')
    if(check_type_image(file.filename) != 'zip'):
        return {file.filename:{"errorCode": 1, "errorMessage": "Invalid .zip file! Please try again.", "data": []}}
    else:
        os.makedirs(pathSave, exist_ok=True)
        async with async_open(os.path.join(pathSave,file.filename),'r') as f:
            await f.write(await file.read())
        with ZipFile(os.path.join(pathSave,file.filename),'r') as zipObj:
            zipObj.extractall('anhtoancanh')
            # Get list of files names in zip
            listOfiles = zipObj.namelist()
            for namefile in listOfiles:
                obj = await asyncio.to_thread(ReturnInfoLP,os.path.join(pathSave,namefile))
                if (obj.errorCode ==0):
                    dict_values.update({namefile:{"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,
                    "data": [{"textPlate": obj.textPlate, "accPlate": obj.accPlate, "imagePlate": 'anhbienso/'+obj.imagePlate}]}})
                else:
                    dict_values.update({namefile:{"errorCode": obj.errorCode, "errorMessage": obj.errorMessage,"data": []}})
            return dict_values
end_time = time.time()
curren_time = end_time-start_time
if __name__ == "__main__":
    uvicorn.run(app,host='127.0.0.2', port=8000)
    print(curren_time)