'''
ib-hussain: I think this file isn'treally necessary but the authors of the papers added it for some reason so I left it in. 
It just calls the training and prediction functions for the weekly model but the train_weeks file or even class is nowhere to be found.
I have added a train_weeks_ibVersion file which has a train_weeks function that is called in this file, but I have no idea if it is the same as the one that the authors of the paper intended to be used here or if it is even correct, but it is something at least.
'''
import os
import dotenv
dotenv.load_dotenv()
import predict_weeks
import train_weeks_ibVersion as train_weeks
# import train_weeks # idk i cant really find the file for this or even the function anywhere so idk what's the purpose of this even but i will implement something for this
# Note: the order of these two calls matters, as the second one relies on the first one to have trained the models and saved them to disk.
train_weeks.main(f"{str(os.getenv('PROCESSOR', 'cpu'))}", f"{str(os.getenv('datasets_out_tickerCollector_path', 'assignment2work/datasetsOut'))}/crypto/daily_20_2190_marked.csv", 104)
predict_weeks.main(f"{str(os.getenv('PROCESSOR', 'cpu'))}", f"{str(os.getenv('datasets_out_tickerCollector_path', 'assignment2work/datasetsOut'))}/crypto/daily_20_2190_marked.csv", 104)
    