export PYTHONPATH=/home/dkai/my_work/programs/meerkats/analyzer/predictor
/home/dkai/anaconda3/envs/python3.8_env/bin/python3 eval.py \
--test_data=/home/dkai/my_work/programs/meerkats/analyzer/predictor/data/val/val.csv \
--batch_size=1 \
--weight=/home/dkai/my_work/programs/meerkats/analyzer/predictor/runs/best.pt \
# --autoregressive \