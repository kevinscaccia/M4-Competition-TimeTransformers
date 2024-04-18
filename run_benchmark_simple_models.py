# python
import argparse
import numpy as np
import pandas as pd
import json
import multiprocessing
# ml
from sklearn.preprocessing import MinMaxScaler
# local
from utils.cnn import SimpleCNN
from experiment import Experiment
from utils.m4 import smape, mase, M4DatasetGenerator
from utils.ml import print_num_weights
import numpy as np
from torch import nn

def naive_predict(train_y, test_y, fh):
    pred_y = np.asarray([train_y[-1]]*fh) 
    return pred_y


class NaivePredictor(nn.Module):
    def __init__(self,):
        super(NaivePredictor, self).__init__()
    
    def is_transformer(self,):
        return False

    def __call__(self, x):
        return x[:,-1,:] # last timestep
    
    def fit(self, conf):
        pass

    def predict(self, x, forecast_horizon):
        y = self(x).repeat((1, forecast_horizon)).unsqueeze(-1)
        # print(y)
        # print(y.shape)
        # raise Exception
        return y
#
# Parse args
#
parser = argparse.ArgumentParser(prog='M4Benchmark', description='Benchmark on M4 Dataset')
parser.add_argument('--model', required=True)
parser.add_argument('--freq', required=True)
args = parser.parse_args()
model_name = args.model
run_sp = args.freq
assert(model_name in ['CNN','Naive','Informer','InformerSmall'])
assert run_sp in ['Hourly','Daily','Weekly','Monthly','Quarterly','Yearly']

prediction_frequency = 'D' #{'Hourly':'h','Daily':'D','Weekly':'W','Monthly':'MS','Quarterly':'QS','Yearly':'YS'}[run_sp]
np.random.seed(123)


def get_model(model_name, model_conf):
    if model_name == 'CNN':
        return SimpleCNN(model_conf['forecasting_horizon'], model_dim=64)
    elif model_name == 'Naive':
        return NaivePredictor()


m4_data = M4DatasetGenerator([run_sp])
num_of_series = m4_data.data_dict[run_sp]['num']
block_size = m4_data.data_dict[run_sp]['fh']
fh = m4_data.data_dict[run_sp]['fh']




def run(proc_args, shared_serie_id, shared_smape, shared_mase):
    train_serie, test_serie, serie_id, fh, freq, serie_sp = proc_args
    assert fh == block_size
    model_conf = {}
    model_conf['input_size'] = min(fh*4, len(train_serie)//10)
    model_conf['forecasting_horizon'] = fh

    model = get_model(model_name, model_conf)
    print_num_weights(model)
    
    exp_conf = {
            # Model
            'model': model,
            'model_n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 
            'input_len':block_size,
            'forecast_horizon':fh,
            'feature_dim':1,
            # Data
            'frequency':serie_sp.lower(),
            'scaler': MinMaxScaler((-1,1)),
            'decompose': False, #detrend and de-sazonalize
            'freq':freq,
            # Others
            'device':'cuda',
            'verbose':False,
    }
    train_conf = {
        'epochs':512,
        'lr':1e-3, 
        'batch_size':512,
        'validate_freq':10,
        'verbose':False,
    }
    exp = Experiment(exp_conf)
    exp.set_dataset(linear_serie=train_serie, train=True)
    # exp.set_dataset(linear_serie=test_serie)
    exp.train(train_conf)
    # test
    last_train_values = train_serie[-block_size:]
    pred_y = exp.predict(last_train_values, fh)
    
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
with multiprocessing.Manager() as manager:
    shared_serie_id, shared_smape, shared_mase = manager.list(),manager.list(),manager.list()
    with manager.Pool(processes=10) as pool:
        pool.starmap(run, [(d, shared_serie_id, shared_smape, shared_mase) for d in all_data])

    metrics_table = {'serie_id':list(shared_serie_id),'smape':list(shared_smape),'mase':list(shared_mase)}

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
    json.dump(metrics_table, open(f'./results/{run_sp}_{model_name}1_metrics_table.json','w'))