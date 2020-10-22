#!/bin/bash
set -x

mkdir -p ./datasets

mkdir -p ./datasets/hp
wget https://zenodo.org/record/4117827/files/hp_train.txt -O ./datasets/hp/train.txt
wget https://zenodo.org/record/4117827/files/hp_test.txt -O ./datasets/hp/test.txt

mkdir -p ./datasets/kindle
wget https://zenodo.org/record/4117827/files/kindle_train.txt -O ./datasets/kindle/train.txt
wget https://zenodo.org/record/4117827/files/kindle_test.txt -O ./datasets/kindle/test.txt

mkdir -p ./datasets/imdb
wget https://zenodo.org/record/4117827/files/imdb_train.txt -O ./datasets/imdb/train.txt
wget https://zenodo.org/record/4117827/files/imdb_test.txt -O ./datasets/imdb/test.txt









