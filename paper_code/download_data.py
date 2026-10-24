from huggingface_hub import snapshot_download
import os

snapshot_download(revision="v1.0-archives", 
    repo_id="legolasflagstaff/RobustGenBench",
    repo_type="dataset",
    local_dir=os.path.expanduser("~/data_processed"),
)
