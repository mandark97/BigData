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


def runs_list():
    runs_arr = []
    for run in runs.aggregate(query):
        run_h = {}
        for artifact in run['artifacts']:
            if artifact['name'] not in ['best_params.json', 'classification_resport.json']:
                continue

            run_h[artifact['name'][:-5]] = json.load(
                gfs.get(artifact['file_id']))

        for metric in run['metrics']:
            run_h[metric['name']] = metric['values'][0]

        run_h = {**run_h,  **run['config']}
        run_h['model'] = run['experiment']['name']

        runs_arr.append(run_h)

    test = pd.DataFrame.from_dict(runs_arr)
    test.to_csv("classification/results/experiments_report.csv", index=False)


def slice_dict(dict, keys):
    return {k: dict[k] for k in keys}


def uniq_configs():
    runs_arr = []
    for run in runs.aggregate(query):
        runs_arr.append(slice_dict(
            run['config'], ['categorical_features', 'numeric_features']))

    df = pd.DataFrame.from_dict(runs_arr)
    df = df.iloc[df.astype(str).drop_duplicates().index]
    df.to_csv("classification/results/unique_configs.csv", index=False)
    df.to_json("classification/results/unique_configs.json", orient='records')


def best_params_per_config():
    runs_arr = []
    for run in runs.aggregate(query):
        run_h = {}
        for artifact in run['artifacts']:
            if artifact['name'] not in ['best_params.json']:
                continue

            run_h[artifact['name'][:-5]] = json.load(
                gfs.get(artifact['file_id']))

        for metric in run['metrics']:
            run_h[metric['name']] = metric['values'][0]

        run_h['model'] = run['experiment']['name']
        run_h = {**run_h, **slice_dict(run['config'], ['categorical_features',
                                                       'numeric_features', 'label', 'dataset'])}
        runs_arr.append(run_h)

    df = pd.DataFrame.from_dict(runs_arr).sort_values(by=['score'])
    df[['model', 'score', 'best_params', 'label', 'numeric_features', 'categorical_features', 'dataset']]\
        .astype(str)\
        .drop_duplicates(subset=['categorical_features', 'numeric_features', 'label', 'dataset', 'model', 'score'])\
        .sort_values(['categorical_features', 'numeric_features', 'label', 'score', 'model'], ascending=[True, True, True, False, True])\
        .to_csv("classification/results/best_params_per_config.csv", index=False)


if __name__ == "__main__":
    runs_list()
    uniq_configs()
    best_params_per_config()
