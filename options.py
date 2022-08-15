import yaml
import os
y = yaml.load(open("options.yaml"), yaml.FullLoader)

root = y["root"]
size = y["size"]
checkpointPath = y["checkpoint"]
model = None
if y["model"] != "None":
    model = y["model"]
outputPath = y["output"]