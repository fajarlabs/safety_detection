from roboflow import Roboflow
rf = Roboflow(api_key="Lw0uMfqptGeE5VKPenjh")
project = rf.workspace("novemberlabs").project("construction-safety-gsnvb-ejchp")
version = project.version(1)
dataset = version.download("yolov11")
                