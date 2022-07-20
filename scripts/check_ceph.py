from petrel_client.client import Client 
import os
import ipdb

petrel_conf = "~/petreloss.conf"
client = Client(petrel_conf)
filename = "1984:s3://openmmlab/datasets/detection/coco/val2017/000000252219.jpg"
bytes = client.Get(filename)