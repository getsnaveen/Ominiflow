FROM python:3.11.11
COPY . /ominiflow/
RUN pip install --upgrade pip 
RUN pip install -r /ominiflow/requirements.txt
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit.run", "/ominiflow/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

