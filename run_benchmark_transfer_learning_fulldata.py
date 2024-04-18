# python
import argparse
import numpy as np
import pandas as pd
import json
# local
from utils.ml import print_num_weights
from utils.transformers import get_model
# Neural models
from neuralforecast.core import NeuralForecast
#
# Parse args
#
parser = argparse.ArgumentParser(prog='M4Benchmark', description='Benchmark on M4 Dataset')
parser.add_argument('--model', required=True)
parser.add_argument('--iterations', required=True)
parser.add_argument('--epochs', required=True)
parser.add_argument('--batch', required=False)
parser.add_argument('--valsteps', required=False)
parser.add_argument('--stop', required=False)
parser.add_argument('--lr', required=False, default=1e-3, type=float)
parser.add_argument('--small', action=argparse.BooleanOptionalAction)

args = parser.parse_args()
model_name = args.model
batch_size = int(args.batch)
lr = float(args.lr)
max_steps = int(args.epochs)
input_factor = 1
early_stop = int(args.stop) 
val_steps = int(args.valsteps)
num_iterations = int(args.iterations)
small = bool(args.small)


assert(model_name in ['VanillaTransformer','Informer','Informer','Autoformer','Autoformer','PatchTST','FEDformer'])

prediction_frequency = 'D' #{'Hourly':'h','Daily':'D','Weekly':'W','Monthly':'MS','Quarterly':'QS','Yearly':'YS'}[run_sp]
np.random.seed(123)

metrics_table = {'serie_id':[],'smape':[],'mase':[],}
smape_list, mase_list = [], []
fh = 48
#
# Train Phase
#
train_all_df = pd.read_parquet('train_all_df.parquet')
input_factor = 1
model_conf = {
        'input_size':fh,
        'input_factor':input_factor,
        'forecasting_horizon': fh,
        'max_steps': max_steps,
        'small': small,
        'batch_size':batch_size,
        'lr':lr,
        'early_stop':early_stop,
        'val_steps':val_steps

    }
print(model_conf)
model = get_model(model_name, model_conf)
print(model_conf)
print_num_weights(model)
nf = NeuralForecast(models=[model],  freq=prediction_frequency, local_scaler_type='standard')


np.random.seed(777)
if small: 
    model_name += 'Small'

for i in range(num_iterations):
    # model train
    nf.fit(df=train_all_df, val_size=64, verbose=True)
    print(f'Loop {i+1}/{num_iterations} OK')
    nf.save(path=f'./trained_models/{model_name}{input_factor}_iter{i}/', model_index=None,  overwrite=True, save_dataset=False)