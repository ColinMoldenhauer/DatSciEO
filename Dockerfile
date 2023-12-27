# this fetches a pre-build base image of pytorch compiled for CUDA > 11.3,
# please use this as the base image for the RS server
FROM rsseminar/pytorch:latest

# install dependencies
RUN pip install numpy
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install tensorboard
RUN pip install torch
RUN pip install torchvision

SHELL ["/bin/bash", "-c"]

# bake repository into dockerfile
RUN mkdir -p ./data
RUN mkdir -p ./models
RUN mkdir -p ./runs
RUN mkdir -p ./utils

ADD data/1123_top10/1123_delete_nan_samples ./data
ADD models ./models
ADD utils ./utils
ADD training.ipynb test.ipynb training.py entry.sh ./
ADD .tmux.conf /root/.tmux.conf

RUN apt-get update && \
  apt install -y tmux

# this is setting your pwd at runtime
WORKDIR /workspace
