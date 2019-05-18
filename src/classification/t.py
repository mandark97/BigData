from sacred.observers import MongoObserver
from sacred import Experiment
from dotenv import load_dotenv

load_dotenv()
ex = Experiment()

ex.observers.append(MongoObserver.create(
    url="mongodb://$MONGO_INITDB_ROOT_USERNAME:$MONGO_INITDB_ROOT_PASSWORD@localhost:27017/?authSource=admin",
    db_name="sacred"
))


@ex.config
def features_config():
    pass


@ex.config
def preproces_config():
    pass


@ex.config
def model_config():
    pass


@ex.automain
def main(_run):
    pass
