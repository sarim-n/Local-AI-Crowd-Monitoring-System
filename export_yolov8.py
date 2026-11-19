import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in .env")

rf = Roboflow(api_key=api_key)
workspace = rf.workspace("sarimahmedn-official-gmail-com")
project = workspace.project("label-cdklr")
version = project.version(1)

print("Starting export and download of dataset in yolov8 format...")
dataset = version.download("yolov8")
print("Dataset downloaded successfully to:", dataset.location)

