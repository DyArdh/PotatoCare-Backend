FROM python:3.9-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
        libfontconfig1 \
        libxrender1 \
        libxcb1 \
        libpng16-16 \
        libtiff6 \ 
        zlib1g \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

ENV KERAS_BACKEND=tensorflow

COPY . .

EXPOSE 8000

CMD ["python", "-m", "app.main"]
