# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

# Install the package and its declared dependencies from pyproject.toml. The source is
# copied first because a PEP 517 build reads the tree it is building, so there is no
# manifest-only layer to install from the way requirements.txt allowed.
RUN python -m pip install .

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python"]
