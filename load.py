import os 
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from tensorboard.plugins import projector
import tensorflow as tf
from db import get_astra
from cassandra.concurrent import execute_concurrent_with_args

use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
FILE_PATH = os.environ.get('FILE_PATH')
df = pd.read_csv(FILE_PATH)
df['embeddings']=use_embed(np.array(df.message)).numpy().tolist()
session, keyspace, table = get_astra()
statement = session.prepare(f'INSERT INTO {keyspace}.{table} (message_id,message,label,message_embedding) values(?,?,?,?)')
parameters = df[["CID","MSG",'label','embeddings']].to_records(index=False).tolist()
execute_concurrent_with_args(session, statement, parameters, concurrency=16)
