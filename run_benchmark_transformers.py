# python
import argparse
import numpy as np
import pandas as pd
import json
import multiprocessing
# local
from utils.m4 import smape, mase, M4DatasetGenerator
from utils.ml import print_num_weights

# Neural models
from neuralforecast.core import NeuralForecast
from neuralforecast.models import Informer, Autoformer, FEDformer, PatchTST, NBEATS, NHITS
#
# Parse args
#
parser = argparse.ArgumentParser(prog='M4Benchmark', description='Benchmark on M4 Dataset')
parser.add_argument('--model', required=True)
parser.add_argument('--freq', required=True)
parser.add_argument('--procs', required=True)
parser.add_argument('--inputfactor', required=True)
parser.add_argument('--epochs', required=False, default=400)

parser.add_argument('--batch', required=False)
parser.add_argument('--lr', required=False, default=1e-3, type=float)
parser.add_argument('--small', action=argparse.BooleanOptionalAction)



args = parser.parse_args()
model_name = args.model
run_sp = args.freq
n_procs = int(args.procs)
max_steps = int(args.epochs)
input_factor = int(args.inputfactor)

batch_size = int(args.batch)
lr = float(args.lr)
small = bool(args.small)



assert run_sp in ['Hourly','Daily','Weekly','Monthly','Quarterly','Yearly']

prediction_frequency = 'D' #{'Hourly':'h','Daily':'D','Weekly':'W','Monthly':'MS','Quarterly':'QS','Yearly':'YS'}[run_sp]
np.random.seed(123)

from utils.transformers import get_model

metrics_table = {'serie_id':[],'smape':[],'mase':[],}
smape_list, mase_list = [], []
m4_data = M4DatasetGenerator([run_sp])
num_of_series = m4_data.data_dict[run_sp]['num']
block_size = m4_data.data_dict[run_sp]['fh']
fh = m4_data.data_dict[run_sp]['fh']



def run(proc_args, shared_serie_id, shared_smape, shared_mase):
    train_serie, test_serie, serie_id, fh, freq, serie_sp = proc_args
    assert fh == block_size
    # synthetic days
    train_daterange = pd.date_range(start='1980', periods=len(train_serie), freq=prediction_frequency)
    test_daterange = pd.date_range(start=train_daterange[-1], periods=len(test_serie)+1, freq=prediction_frequency)[1:] # len + 1 because the first day is on train dates
    #
    model_conf = {}
    model_conf = {
        'input_size':fh,
        'input_factor':input_factor,
        'forecasting_horizon': fh,
        'max_steps': max_steps,
        'small': small,
        'batch_size':batch_size,
        'lr':lr,
    }
    model = get_model(model_name, model_conf)
    print_num_weights(model)
    
    train_daterange = pd.date_range(start='1980', periods=len(train_serie), freq=prediction_frequency)
    test_daterange = pd.date_range(start=train_daterange[-1], periods=len(test_serie)+1, freq=prediction_frequency)[1:] # len + 1 because the first day is on train dates
    nf = NeuralForecast(models=[model], freq=prediction_frequency, local_scaler_type='standard')
    train_df = pd.DataFrame({
        'unique_id':serie_id,
        'y':train_serie, 
        'ds':train_daterange
        })
    val_size = 0#int(.1 * len(train_serie)) # 20% for validation

    # model train
    nf.fit(df=train_df, val_size=val_size, verbose=False)
    pred_y = nf.predict()
    #
    assert all(pred_y.ds == test_daterange) # check 
    pred_y = pred_y[model_name.replace('Small','')].values

    
    # check if negative or extreme (M4)
    for i in range(len(pred_y)):
        if pred_y[i] < 0:
            pred_y[i] = 0
                
        if pred_y[i] > (1000 * max(train_serie)):
            pred_y[i] = max(train_serie)
    # Metrics
    serie_smape = float(smape(test_serie, pred_y)*100)
    serie_mase = float(mase(train_serie, test_serie, pred_y, freq))
    shared_serie_id.append(serie_id)
    shared_smape.append(serie_smape)
    shared_mase.append(serie_mase)
    print(f'Serie {serie_id}-{serie_sp} Finished')




all_data = m4_data.get(random=False)
shared_serie_id, shared_smape, shared_mase = list(),list(),list()
for d in all_data:
    run(d, shared_serie_id, shared_smape, shared_mase)

metrics_table = {'serie_id':shared_serie_id,'smape':shared_smape,'mase':shared_mase}

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
json.dump(metrics_table, open(f'./single_serie_results/{run_sp}_{model_name}_metrics_table.json','w'))
# with multiprocessing.Manager() as manager:
#     shared_serie_id, shared_smape, shared_mase = manager.list(),manager.list(),manager.list()
#     with manager.Pool(processes=n_procs) as pool:
#         pool.starmap(run, [(d, shared_serie_id, shared_smape, shared_mase) for d in all_data])

#     metrics_table = {'serie_id':list(shared_serie_id),'smape':list(shared_smape),'mase':list(shared_mase)}

#     metrics_dict = {
#         'smape_mean': np.round(np.mean(metrics_table['smape'], dtype=float), 3), 
#         'mase_mean':  np.round(np.mean(metrics_table['mase'], dtype=float), 3),
#         #
#         'smape_std':  np.round(np.std(metrics_table['smape'], dtype=float), 3),
#         'mase_std':   np.round(np.std(metrics_table['mase'], dtype=float), 3),
#     }
#     print(f'''
#         Experiment Finished
#     ''')
#     for k, v in metrics_dict.items(): print(f'      {k}: {v}')
#     json.dump(metrics_table, open(f'./single_serie_results/{run_sp}_{model_name}_metrics_table.json','w'))