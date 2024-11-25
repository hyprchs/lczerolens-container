# Use a slim Python base image
FROM python:3.9-slim

# Set up a working directory
WORKDIR /app

# Install Poetry and system dependencies required for building Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry \
    && apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Also install jupyter
    && pip install --no-cache-dir jupyter

# Expose the port for Jupyter
EXPOSE 8888

# Copy dependency files to lczerolens directory
# COPY lib/lczerolens/pyproject.toml lib/lczerolens/poetry.lock /app/lib/lczerolens/

# Install dependencies using Poetry
WORKDIR /app/lib/lczerolens
# RUN poetry install --with dev,demo --no-root

# Copy the rest of the library files to the container
COPY lib/lczerolens /app/lib/lczerolens

# TODO Organize these layers better
# Install the library itself in editable mode
RUN pip install -e .

# Install lc0 python bindings
RUN pip install lczero-bindings

RUN pip install "numpy<2.0"

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Set up the working directory for your project
WORKDIR /app

# Start Jupyter Notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--notebook-dir=/app", "--NotebookApp.token=''", "--NotebookApp.password=''"]
