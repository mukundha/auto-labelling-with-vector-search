### Automatic Labelling of Training data with Vector search

Start with a dataset in this format,

|message_id|message|label|
|--|--|--|
|1|This is a good movie|Positive|
|2|This is a bad movie|Negative|


Setup the environment

```
export FILE_PATH="source/data.csv"
export OPENAI_API_KEY=
export ASTRA_DB_SECURE_BUNDLE_PATH=
export ASTRA_DB_APPLICATION_TOKEN=
export ASTRA_DB_KEYSPACE="demo"
export ASTRA_DB_TABLE="messages_table"
```


Load data to Astra DB 

```
CREATE TABLE IF NOT EXISTS messages_table (
    message_id int PRIMARY KEY,
    message text,
    label text,
    message_embedding vector<float, 512>
);

CREATE CUSTOM INDEX IF NOT EXISTS message_embedding_index ON demo.messages_table (message_embedding) USING 'org.apache.cassandra.index.sai.StorageAttachedIndex';

CREATE CUSTOM INDEX IF NOT EXISTS label_index ON demo.messages_table (label) USING 'org.apache.cassandra.index.sai.StorageAttachedIndex';
```

```
python3 load.py
```

Predict 

```
python3 predict.py
```
