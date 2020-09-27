import train
import zipfile
import os
import sys
import shutil

models = ["lstm", "mlp", "covNet"]
layers = [
            [64, 128, 256], 
            [[64], [128], [256]],
            [[64, 128], [128, 256], [32, 64]]
        ]

#train_80_path = 'Data/Train/Train_80_Percent'
#val_10_path = 'Data/Validation/Validation_10_Percent'
#test_10_path = 'Data/Test/Test_10_Percent'
#train_10min_path = 'Data/Train/Under_10_min_training'
#tune_90min_path = 'Data/Train/Under_90_min_tuning'
print(sys.argv[1])

with zipfile.ZipFile(sys.argv[1],"r") as zip_ref:
    zip_ref.extractall(".")
os.rename("Temp/processed_data.npy","Temp/Under_90_min_tuning.npy")


with zipfile.ZipFile(sys.argv[2],"r") as zip_ref:
    zip_ref.extractall(".")
os.rename("Temp/processed_data.npy","Temp/Validation_10_Percent.npy")


f = open('tuning_results.txt', 'w')

best_rmse = 999999
best_model = ""
best_hyperparameter = ""
for i in range(len(models)):
    for j in range(len(layers[i])):
        print("Model ",models[i],j)
        rmse = train.training(models[i], layers[i][j], 0)
        
        f.write('{}, {}, {}\n'.format(models[i], layers[i][j], rmse))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = models[i]
            best_hyperparameter = layers[i][j]
            
            
f.close()


f = open('hyperparameter.txt', 'w')
f.write('{}, {}, {}\n'.format(best_model, best_hyperparameter, best_rmse))
f.close()



