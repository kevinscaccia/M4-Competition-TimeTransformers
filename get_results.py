# python
import argparse
import numpy as np
import pandas as pd
import json
# local
from utils.m4 import smape, mase, M4DatasetGenerator
# Neural models
from neuralforecast.core import NeuralForecast
#
# Parse args
#
parser = argparse.ArgumentParser(prog='Get Results', description='Get results M4 Dataset')
parser.add_argument('--model', required=True)
parser.add_argument('--small', action=argparse.BooleanOptionalAction)

args = parser.parse_args()
model_name = args.model
small = bool(args.small)


assert(model_name in ['VanillaTransformer','Informer','Informer','Autoformer','Autoformer','PatchTST','FEDformer'])
if small:
    model_name += 'Small'
input_factor = 1
iter = 0

train_all_df = pd.read_parquet('train_all_df.parquet')
nf = NeuralForecast.load(path=f'./trained_models/{model_name}{input_factor}_iter{iter}/')
pred_y_all_predictions = nf.predict(train_all_df)

full_smapes = []
full_mases = []
for run_sp in ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']:
    m4_data = M4DatasetGenerator([run_sp])
    #
    metrics_table = {'serie_id':list(),'smape':list(),'mase':list()}

    for train_serie, test_serie, serie_id, fh, freq, serie_sp in m4_data.get(random=False):    
        pred_y = pred_y_all_predictions.loc[serie_id][model_name.replace('Small','')].values
        #
        pred_y = pred_y[:fh] # transfered window is less or equal
        #
        serie_smape = smape(test_serie, pred_y)*100
        serie_mase = mase(train_serie, test_serie, pred_y, freq)
        #
        metrics_table['serie_id'].append(serie_id)
        metrics_table['smape'].append(float(serie_smape))
        metrics_table['mase'].append(float(serie_mase))
        full_smapes.append(float(serie_smape))
        full_mases.append(float(serie_mase))

    metrics_dict = {
        'smape_mean': np.round(np.mean(metrics_table['smape'], dtype=float), 3), 
        'mase_mean':  np.round(np.mean(metrics_table['mase'], dtype=float), 3),
    }
    print(run_sp)
    for k, v in metrics_dict.items(): print(f'      {k}: {v}')
metrics_dict = {
        'smape_mean': np.round(np.mean(full_smapes, dtype=float), 3), 
        'mase_mean':  np.round(np.mean(full_mases, dtype=float), 3),
    }
print('--- final results:')
for k, v in metrics_dict.items(): print(f'      {k}: {v}')