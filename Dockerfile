FROM python:3.10 

# We can just use root in build stage since we will be copying files
# over in next stage and reconfiguring permissions
USER root

# Set the working directory in the container
WORKDIR /app

RUN pip install uv

# Copy necessary files for dependency installation
COPY uv.lock pyproject.toml .marimo.toml README.md /app/

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy the source code into the image
ADD src /app/src

# Define environment variables here ...
ENV LOG_LEVEL=INFO
ENV PORT=8000

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev

CMD uv run --frozen --no-dev marimo run src/main.py --host 0.0.0.0 --port 8000 --session-ttl=0