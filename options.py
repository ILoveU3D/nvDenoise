import yaml
import os
y = yaml.load(open("options.yaml"), yaml.FullLoader)

trainRoot = y["root"][0]
validRoot = y["root"][1]
size = y["size"]
checkpointPath = y["checkpoint"]
model = None
if y["model"] != "None":
    model = y["model"]
outputPath = y["output"]