# Start from the official Python base image.
#FROM nexus.bmwgroup.net/python:3.10
FROM python:3.10

# This is where we'll put the requirements.txt file and the code directory.
WORKDIR /code

# Copy the file with the requirements to the /code directory.
COPY ./__init__.py /code/__init__.py
COPY ./requirements.txt /code/requirements.txt
COPY ./final_models /code/final_models
copy ./data/processed /code/data/processed 
COPY ./images /code/images
COPY ./src/__init__.py /code/src/__init__.py
COPY ./src/config.yaml /code/src/config.yaml
COPY ./src/utils.py /code/src/utils.py
COPY ./src/data_preparation_pipeline /code/src/data_preparation_pipeline
COPY ./src/data_preprocessing_pipeline /code/src/data_preprocessing_pipeline
COPY ./src/deployment /code/src/deployment

# Install the package dependencies in the requirements file.
#RUN apt-get install -y git
RUN apt-get update && apt-get install -y git
RUN pip install -i https://nexus.bmwgroup.net/repository/pypi/simple -r /code/requirements.txt

#RUN python /code/setup.py install

ENV PYTHONPATH "${PYTHONPATH}:/code/src"

# Start API
ENTRYPOINT ["streamlit", "run", "src/deployment/website.py", "--server.port=7071", "--server.address=0.0.0.0"]
