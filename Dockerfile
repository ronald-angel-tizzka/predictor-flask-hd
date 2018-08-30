# Downloading latest Condas image from Repository.
FROM continuumio/miniconda3:latest

# Defining working Folder.
WORKDIR /app

# Install classifier_env Requirements.
COPY environment.yml /app/environment.yml

# Creating Condas Environment.
RUN conda env create -n classifier_env -f environment.yml

# Install Application withing Container.
COPY . /app/

# Activate the classifier_env Environment.
ENV PATH /opt/conda/envs/classifier_env/bin:$PATH

# Adding command to deploy service.
CMD python -u service.py