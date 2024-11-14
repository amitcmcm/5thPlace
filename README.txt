main_clean_mlflow: main training file


.................preprocessing: ...................
currently, there are preprocessed data in the folder "data_np". 

if you want to preprocess the data yourself, you will need to run "data_creation_clean_2d57c2_removed" before running the training code. 
(the removed subject, 2d57c2, was causing bugs, you can delete the part in the code that removes it and debug if you don't want to exclude it)
data_creation_clean takes the datasets you select (using the dictionary INPUT_DATA at the start of the code) and does the final stage of preprocessing, saving all the files in data_np. it also prepares an "internal" test set out of the training data. 
- if you want to preprocess either one of the daily living datasets yourself, please read "README_daily_living_data.txt". you will have to run data_creation_clean again afterwards. 
- delete the existing content of the folders before preprocessing (i suggest saving backup of them first). the code doesn't delete them automatically. 


.................inference and testing: ...................
preprocess_localtest: run this after data_creation_clean. preprocesses the internal test data and saves in the "test" folder. 
inference_clean_localtest: inference on the internal test data. you need to select your model checkpoints, i saved mine under their mlflow run names, hence the input variable MLFLOW_RUN_NAME. the code saves a "submission" file that can be uploaded to kaggle for scoring or analyzed using analyze_test. 
