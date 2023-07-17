# Start from the official Python base image.
FROM nexus.bmwgroup.net/python:3.11

# This is where we'll put the requirements.txt file and the app directory.
WORKDIR /code

# Copy the file with the requirements to the /code directory.
COPY ./requirements.txt /code/requirements.txt
COPY ./src /code/src
COPY ./models /code/models
COPY ./__init__.py /code/__init__.py

# Install the package dependencies in the requirements file.
RUN apt-get install -y git
RUN pip install -i https://nexus.bmwgroup.net/repository/pypi/simple -r /code/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/code/src"

# Start API
CMD ["uvicorn", "src.api.api:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "7070"]
