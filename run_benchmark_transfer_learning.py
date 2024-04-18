# python
import argparse
import numpy as np
import pandas as pd
import json
# local
from utils.m4 import smape, mase, M4DatasetGenerator
from utils.ml import print_num_weights
from utils.transformers import get_model
# Neural models
from neuralforecast.core import NeuralForecast
#
# Parse args
#
parser = argparse.ArgumentParser(prog='M4Benchmark', description='Benchmark on M4 Dataset')
parser.add_argument('--model', required=True)
parser.add_argument('--freq', required=True)
parser.add_argument('--epochs', required=False, default=400)
parser.add_argument('--batch', required=False, default=1024)
parser.add_argument('--lr', required=False, default=1e-3, type=float)
parser.add_argument('--inputfactor', required=True)
parser.add_argument('--small', action=argparse.BooleanOptionalAction)
args = parser.parse_args()
model_name = args.model
run_sp = args.freq
batch_size = int(args.batch)
lr = float(args.lr)
max_steps = int(args.epochs)
input_factor = int(args.inputfactor)
small = bool(args.small)


assert(model_name in ['VanillaTransformer','Informer','Informer','Autoformer','Autoformer','PatchTST','FEDformer'])
assert run_sp in ['Hourly','Daily','Weekly','Monthly','Quarterly','Yearly']
prediction_frequency = 'D' #{'Hourly':'h','Daily':'D','Weekly':'W','Monthly':'MS','Quarterly':'QS','Yearly':'YS'}[run_sp]
np.random.seed(123)

metrics_table = {'serie_id':[],'smape':[],'mase':[],}
smape_list, mase_list = [], []
m4_data = M4DatasetGenerator([run_sp])
num_of_series = m4_data.data_dict[run_sp]['num']
block_size = m4_data.data_dict[run_sp]['fh']
fh = m4_data.data_dict[run_sp]['fh']
#
# Train Phase
#
train_all_df = []
test_all_df = []
for train_serie, test_serie, serie_id, fh, freq, serie_sp in m4_data.get(random=False):
    train_daterange = pd.date_range(start='1800', periods=len(train_serie), freq=prediction_frequency)
    test_daterange = pd.date_range(start=train_daterange[-1], periods=len(test_serie)+1, freq=prediction_frequency)[1:] # len + 1 because the first day is on train dates
    train_all_df.append(pd.DataFrame({
        'unique_id':[serie_id]*len(train_serie),
        'ds':train_daterange,
        'y':train_serie
    }))
    test_all_df.append(pd.DataFrame({
        'unique_id':[serie_id]*len(test_serie),
        'ds':test_daterange,
        'y':test_serie
    }))
train_all_df = pd.concat(train_all_df)
test_all_df = pd.concat(test_all_df).set_index('unique_id')

model_conf = {
        'input_size':fh,
        'input_factor':input_factor,
        'forecasting_horizon': fh,
        'max_steps': max_steps,
        'small': small,
        'batch_size':batch_size,
        'lr':lr
    }
print(model_conf)
model = get_model(model_name, model_conf)
print(model_conf)
print_num_weights(model)
# model train
nf = NeuralForecast(models=[model], freq=prediction_frequency, local_scaler_type='standard')
nf.fit(df=train_all_df, val_size=0, verbose=False)
# 
# Eval
#
metrics_table = {'serie_id':list(),'smape':list(),'mase':list()}

pred_y_all_predictions = nf.predict(train_all_df)

for train_serie, test_serie, serie_id, fh, freq, serie_sp in m4_data.get(random=False):    
    pred_y = pred_y_all_predictions.loc[serie_id][model_name].values
    serie_smape = smape(test_serie, pred_y)*100
    serie_mase = mase(train_serie, test_serie, pred_y, freq)
    #
    metrics_table['serie_id'].append(serie_id)
    metrics_table['smape'].append(float(serie_smape))
    metrics_table['mase'].append(float(serie_mase))

metrics_dict = {
    'smape_mean': np.round(np.mean(metrics_table['smape'], dtype=float), 3), 
    'mase_mean':  np.round(np.mean(metrics_table['mase'], dtype=float), 3),
    #
    'smape_std':  np.round(np.std(metrics_table['smape'], dtype=float), 3),
    'mase_std':   np.round(np.std(metrics_table['mase'], dtype=float), 3),
}
print(f'''
    Experiment Finished
''')
for k, v in metrics_dict.items(): print(f'      {k}: {v}')
if small:
    model_name += 'Small'
json.dump(metrics_table, open(f'./results/{run_sp}_{model_name}{input_factor}_TF_metrics_table.json','w'))
nf.save(path=f'./checkpoints/{run_sp}_{model_name}{input_factor}/', model_index=None,  overwrite=True, save_dataset=False)