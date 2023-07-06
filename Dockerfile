# Start from the official Python base image.
FROM python:3.9

# This is where we'll put the requirements.txt file and the app directory.
WORKDIR /code

# Copy the file with the requirements to the /code directory.
COPY ./requirements.txt /code/requirements.txt
COPY ./src /code/src
COPY ./models /code/models
COPY ./__init__.py /code/__init__.py

# Install the package dependencies in the requirements file.
RUN apt-get update && apt-get install -y git
RUN pip install --trusted-host pypi.python.org -r /code/requirements.txt

# 
CMD ["uvicorn", "src.api.api:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
