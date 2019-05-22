import json

import pandas as pd
from gridfs import GridFS
from pymongo import MongoClient

client = MongoClient("localhost", 27017)
db = client['sacred']
runs = db['runs']
gfs = GridFS(db)

query = [
    {
        "$match": {
            "status": "COMPLETED"
        }
    },
    {
        "$lookup": {
            "from": "metrics",
            "localField": "_id",
            "foreignField": "run_id",
            "as": "metrics"
        }
    },
    {
        "$project": {
            "_id": 0,
            "metrics.name": 1,
            "metrics.values": 1,
            "artifacts": 1,
            "config": 1,
            "experiment.name": 1
        }
    },
    {
        "$sort": {
            "experiment.name": 1
        }
    }
]


runs_arr = []
for run in runs.aggregate(query):
    run_h = {}
    for artifact in run['artifacts']:
        if artifact['name'] in ['model.joblib', 'cv_results.csv']:
            continue

        run_h[artifact['name'][:-5]] = json.load(
            gfs.get(artifact['file_id']))

    for metric in run['metrics']:
        run_h[metric['name']] = metric['values'][0]

    run_h['config'] = run['config']
    run_h['model'] = run['experiment']['name']

    runs_arr.append(run_h)


test = pd.DataFrame.from_dict(runs_arr)
test.to_csv("experiments_report.csv", index=False)
