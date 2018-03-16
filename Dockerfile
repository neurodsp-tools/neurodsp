FROM continuumio/miniconda3

# conda doesn't support /bin/sh
ENTRYPOINT ["/bin/bash", "-c"]
EXPOSE 5555

# Get the image up to date and create a working directory
RUN apt-get update -qq && apt-get upgrade -qq -y && conda update conda -q -y
WORKDIR /neurodsp

# Create the conda environment and install dependencies
COPY requirements.txt /neurodsp/requirements.txt
COPY setup.py /neurodsp/setup.py
RUN conda create -n neurodsp python -q -y
RUN ["/bin/bash", "-c", "source activate neurodsp && \
                        pip install -q jupyter && \
                        pip install -q -r requirements.txt && \
                        python setup.py -q develop"]


# If not in interactive mode, make the tutorials available over the exposed port
COPY . /neurodsp
CMD ["source activate neurodsp && jupyter notebook tutorials/ --ip=0.0.0.0 --port=5555 --no-browser --allow-root"]
