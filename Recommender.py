from surprise import Dataset, Reader, accuracy
from surprise import SVD, evaluate, SVDpp
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split
import numpy as np
import pandas as pd
import time
import csv
from surprise.dump import dump, load

start_time = time.time()

print('-- Loading data into initial DataFrame --')
datadf = pd.read_json('./reviews.training.json', lines=True)

print('-- Loading Training Data Into Data Object --')
reader = Reader(rating_scale=(1, 5))
datadf_training = datadf[['reviewerID', 'asin', 'overall']]
print(datadf_training.head())
data = Dataset.load_from_df(datadf_training, reader=reader)
trainset = data.build_full_trainset()

print('-- Loading Algo --')
algo_svd = SVD(n_factors=10, n_epochs=100, lr_all=0.005, reg_all=0.2)

print('-- Training Models --')
algo_svd.fit(trainset)
print('-- SVD Fitting Done --')

print('Done ' + "--- %s seconds ---" % (time.time() - start_time))

print('-- Writing CSV --')
def svd_outputs(algo):
    with open('reviews.test.unlabeled.csv', 'r') as test:
        svd_list=[]
        for row in test.readlines()[1:]:
            dataid, reviewid, asin = row.split(',')
            overall = algo.predict(str(reviewid).strip(), str(asin).strip())
            svd_list.append({'datapointID': dataid, 'overall':overall.est})
    test.close()
    print(svd_list[:5])
    with open('outputsSvd.csv', 'w') as out:
        fieldnames = ['datapointID', 'overall']
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for each in svd_list:
            writer.writerow(each)
    out.close()

## Uncomment to get the CSV file for Kaggle submission
# svd_outputs(algo_svd)

def svd_evaluation(algo):
    with open('reviews.dev.csv', 'r') as dev:
        svd_list=[]
        for row in dev.readlines()[1:]:
            reviewid, asin, overall1 = row.split(',')
            overall2 = algo.predict(reviewid, str(asin))
            svd_list.append({'reviewID':reviewid,
                             'asin': str(asin),
                             'init_overall':overall1.strip('\n'), 
                             'pred_overall':overall2.est})
    dev.close()
            
    with open('evaluateSvd.csv', 'w') as out:
        fieldnames = ['reviewID', 'asin', 'init_overall', 'pred_overall']
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for each in svd_list:
            writer.writerow(each)
    out.close()

## Uncomment to get the evaluation CSV file used for error analysis
# svd_evaluation(algo_svd)

# Last ditch effort as a sanity check
def test_accuracy():
    datadf = pd.read_json('./reviews.training.json', lines=True)
    print('-- Loading Training Data Into Data Object --')
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(datadf[['reviewerID', 'asin', 'overall']], reader=reader)
    print('-- Loading Algo --')
    trainset, testset = train_test_split(data, test_size=.25)
    algo_svd = SVD(n_factors=10, n_epochs=100, lr_all=0.005, reg_all=0.2)
    algo_svd.fit(trainset)
    predictions = algo_svd.test(testset)
    print(accuracy.rmse(predictions))
    
## Uncomment to test the accuracy of the model on a train/test split set
# test_accuracy()