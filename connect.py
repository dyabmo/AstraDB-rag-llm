from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access variables
ASTRADB_ID = os.getenv('ASTRADB_ID')
ASTRADB_SECRET = os.getenv('ASTRADB_SECRET')

cloud_config= {
  'secure_connect_bundle': 'secure-connect-demo-llm.zip'
}
auth_provider = PlainTextAuthProvider(ASTRADB_ID, ASTRADB_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

row = session.execute("select release_version from system.local").one()
if row:
  print(row[0])
else:
  print("An error occurred.")