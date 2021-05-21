#!/bin/bash

requirements:
	pip install -r requirements.txt
	
run:
	python3 kmeans.py >> output.txt

clean:
	> output.txt
