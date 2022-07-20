import json 
import ipdb

val_anno = "../data/annotations/instances_val2017.json"
new_anno = "../data/annotations/val_mini.json"

new_anno_dict = {}

with open(val_anno, "r") as f:
    annos = json.load(f)
    
    new_anno_dict["info"] = annos["info"]
    new_anno_dict["licenses"] = annos["licenses"]
    new_anno_dict["images"] = annos["images"][:100]
    new_anno_dict["annotations"] = annos["annotations"]
    new_anno_dict["categories"] = annos["categories"] 

with open(new_anno, "w") as g:
    json.dump(new_anno_dict, g)
g.close()