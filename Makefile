PYTHON = python

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  init           to install required packages"
	@echo "  build          to build the python package(s)"
	@echo "  install        to build and install the python package(s)"
	@echo "  develop        to build and install the python package(s) for development"
	@echo "  test           to run all integration and unit tests"
	@echo "  docker         to build the Docker image and launch the tutorials at localhost:5555"
	@echo ""
	@echo "Advanced targets"
	@echo "  docker-build   to build only the Docker image from Dockerfile"
	@echo "  docker-clean   to delete all generated or dangling Docker images"
	@echo "  docker-run     to launch Docker container with the neurodsp:latest ID"
	@echo "  docker-stop    to halt a running neurodsp container"

init:
	pip install -r requirements.txt

build:
	$(PYTHON) setup.py build

install: build
	$(PYTHON) setup.py install

develop: build
	$(PYTHON) setup.py develop

test: develop
	pytest

docker-build:
	@echo "Creating Docker image - neurodsp:latest"
	docker build . -t neurodsp

docker-clean:
	@echo "Removing latest and dangling images"
	docker rmi neurodsp:latest

docker-run:
	@echo "Launching container - neurodsp"
	docker run -d -it -p 5555:5555 --rm --name neurodsp neurodsp:latest
	@echo "Copy and paste this URL into your browser to access notebooks"
	@sleep 1
	@docker logs neurodsp | grep token -m 1

docker-stop:
	@echo "Shutting down container"
	docker stop neurodsp

docker: docker-build docker-run
