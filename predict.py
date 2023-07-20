import os 
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from tensorboard.plugins import projector
import tensorflow as tf
from db import get_astra
from cassandra.concurrent import execute_concurrent_with_args
import concurrent.futures

df = pd.read_csv('source/input.csv')
session, keyspace, table = get_astra()
embedding_column = 'message_embedding'
unique_labels =  df.label.unique()
unique_labels = unique_labels[unique_labels!='Unlabelled']
labels = tuple(unique_labels)

def predict_label(row):
    statement = f"SELECT label FROM {keyspace}.{table} WHERE label IN {labels} ORDER BY {embedding_column} ANN of  {row['embeddings']} LIMIT 5"    
    rs = session.execute(statement)
    results = rs._current_rows
    if len(results) > 0:
        first = results[0].label
        same_label = all( r.label== first for r in results)
        if same_label:
            return (row['message_id'],row['message'],first)
        else:            
            print(f'Multiple labels for {row["message_id"]}, {[r.label for r in results]}')
    return (row['message_id'],row['message'],'Unlabelled')


with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(predict_label,row) for _,row in df[df.label=='Unlabelled'].iterrows()]
    results = [future.result() for future in concurrent.futures.as_completed(futures)]

results_df = pd.DataFrame(results,columns=['message_id','message','label'])
results_df.to_csv('results.csv', index=False)

# OPTIONAL: To update the labels
# def update_labels(row):
#     statement = f"UPDATE {keyspace}.{table} SET label='{row.label}' where message_id={row.message_id}"
#     session.execute(statement)
#     df.label[df.message_id==row.message_id]=row.label
#     return

# with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
#     futures = [executor.submit(update_labels,row) for _,row in results_df[results_df.label != 'Unlabelled'].iterrows()]
#     update_results = [future.result() for future in concurrent.futures.as_completed(futures)]

# df.to_csv('updated_messages.csv', index=False)




