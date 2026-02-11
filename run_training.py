import subprocess

subprocess.run(["python","src/preprocess.py"])
subprocess.run(["python","src/train.py"])
subprocess.run(["python","src/evaluate.py"])
subprocess.run(["python","src/export.py"])