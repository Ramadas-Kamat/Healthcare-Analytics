#!/usr/bin/env bash

MIMIC_DATA='./sub-dataset'
PROJECT='.'

docker run -it -d --privileged=true \
  --cap-add=SYS_ADMIN \
  -m 8192m -h bootcamp.local \
  --name bigbox -p 2222:22 -p 9530:9530 -p 8888:8888\
  -v ${MIMIC_DATA}/:/mnt/data \
  -v ${PROJECT}/:/mnt/host \
  sunlab/bigbox:latest \
  /scripts/entrypoint.sh