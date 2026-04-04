'''
ib-hussain: I think this file isn't really necessary but the authors of the papers added it for some reason so I left it in. 
It just calls the training and prediction functions for the weekly model but the train_weeks file or even class is nowhere to be found.
I have added a train_weeks_ibVersion file which has a train_weeks function that is called in this file, but I have no idea if it is the same as the one that the authors of the paper intended to be used here or if it is even correct, but it is something at least.

(Added by AI Agent)
Module: run.py
This script sequentially executes the training and then the prediction process for the weekly models.
'''
import os
import dotenv
dotenv.load_dotenv()
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations for CPU
# Enable XLA compilation for faster CPU execution
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
# Set CPU affinity for better performance
try:
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() - 1)  # Leave 1 core for system
    tf.config.threading.set_inter_op_parallelism_threads(2)  # For parallel ops
except:
    pass
import predict_weeks
import train_weeks_ibVersion as train_weeks
# import train_weeks # idk i cant really find the file for this or even the function anywhere so idk what's the purpose of this even but i will implement something for this
# Note: the order of these two calls matters, as the second one relies on the first one to have trained the models and saved them to disk.
train_weeks.main(device = f"{str(os.getenv('PROCESSOR', 'cpu'))}", data_path=f"{str(os.getenv('datasets_out_tickerCollector_path', 'assignment2work/datasetsOut'))}/crypto/daily_20_2190_marked.csv", starting_week=2, num_weeks=21, horizon=5)
print("Training completed")
predict_weeks.main(device = f"{str(os.getenv('PROCESSOR', 'cpu'))}", data_path=f"{str(os.getenv('datasets_out_tickerCollector_path', 'assignment2work/datasetsOut'))}/crypto/daily_20_2190_marked.csv", starting_week=2, num_weeks=21, horizon=7)

# Lubabah, run this command:
# python assignment2work/MTGNN/run.py > assignment2work/MTGNN/run_results.txt
    