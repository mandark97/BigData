# BigData

## Dataset

https://www.kaggle.com/uciml/student-alcohol-consumption

## Sacred docker setup

### Sacred

Sacred is a tool to help you configure, organize, log and reproduce experiments developed at IDSIA.
https://sacred.readthedocs.io/en/latest/

The logs are stored in MongoDB.

### Requirements

- [docker](https://www.docker.com/)
- [docker-compose](https://docs.docker.com/compose/)

Run `docker-compose up --build`
Open omniboard (`http://localhost:9000`) to see the experiment results

## Mongo db dump/restore

### Dump

To dump mongodb to a file use

`docker exec big_data_mongo sh -c 'exec mongodump -d sacred --archive' > classification/results/results.archive`

### Restore

To restore the data from the dump file you have to:

- copy the archive in the container

`docker cp classification/results/results.archive big_data_mongo:/`

- run mongorestore

`docker exec big_data_mongo sh -c 'exec mongorestore -d sacred --archive=results.archive'`
