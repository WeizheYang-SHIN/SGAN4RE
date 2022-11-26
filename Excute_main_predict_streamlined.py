import os
import datetime
start_time = datetime.datetime.now()
print('Start excuting main_predict_streamlined.py')
while True:
    end_time = datetime.datetime.now()
    if (end_time - start_time).seconds/60>=1.5:

        start_time = datetime.datetime.now()
        check_model_name = open('./output/models/checkpoint', 'r', encoding='utf-8').readline()
        print('Start predicting with model----------' + str(check_model_name))
        os.system('python main_predict_streamlined.py')
