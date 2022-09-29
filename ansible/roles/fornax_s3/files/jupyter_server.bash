set -e
bash
/home/ubuntu/mambaforge/bin/mamba init
. /home/ubuntu/.bashrc
mamba activate fornax-bench
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0
