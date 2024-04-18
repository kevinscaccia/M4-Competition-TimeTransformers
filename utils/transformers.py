from neuralforecast.models import VanillaTransformer, Informer, Autoformer, FEDformer, PatchTST
def get_model(model_name, model_conf):
    input_size = model_conf['input_size'] * model_conf['input_factor']
    batch_size = model_conf['batch_size']
    lr = model_conf['lr']
    early_stop = model_conf.get('early_stop', -1)
    val_steps = model_conf.get('val_steps', 1024)
    if model_name == 'VanillaTransformer':
        if model_conf['small']:
            return VanillaTransformer(
                    input_size=input_size, h=model_conf['forecasting_horizon'], scaler_type='standard', 
                    hidden_size=16, conv_hidden_size=32, n_head=2, 
                    max_steps=model_conf['max_steps'], batch_size=batch_size, learning_rate=lr, early_stop_patience_steps=early_stop, val_check_steps=val_steps)
        else:
            return VanillaTransformer(
                    input_size=input_size, h=model_conf['forecasting_horizon'], scaler_type='standard',  
                    hidden_size=64, conv_hidden_size=32, n_head=4, 
                    max_steps=model_conf['max_steps'], batch_size=batch_size, learning_rate=lr, early_stop_patience_steps=early_stop, val_check_steps=val_steps)
    
    if model_name == 'Informer': 
        if model_conf['small']:
            return Informer(
                    input_size=input_size, h=model_conf['forecasting_horizon'], scaler_type='standard', 
                    hidden_size=16, n_head=2, conv_hidden_size=8, 
                    max_steps=model_conf['max_steps'], batch_size=batch_size, learning_rate=lr, early_stop_patience_steps=early_stop, val_check_steps=val_steps)
        else: 
            return Informer(
                    input_size=input_size, h=model_conf['forecasting_horizon'], scaler_type='standard',  
                    hidden_size=64, n_head=4, conv_hidden_size=8, 
                    max_steps=model_conf['max_steps'], batch_size=batch_size, learning_rate=lr, early_stop_patience_steps=early_stop, val_check_steps=val_steps)
    
    if model_name == 'Autoformer':
        if model_conf['small']:
            return Autoformer(
                    input_size=input_size, h=model_conf['forecasting_horizon'], scaler_type='standard', 
                    hidden_size = 16, conv_hidden_size = 32, n_head=2, 
                    max_steps=model_conf['max_steps'], batch_size=batch_size, learning_rate=lr, early_stop_patience_steps=early_stop, val_check_steps=val_steps)
        else:
            return Autoformer(
                    input_size=input_size, h=model_conf['forecasting_horizon'], scaler_type='standard',  
                    hidden_size = 64, conv_hidden_size = 32, n_head=4, 
                    max_steps=model_conf['max_steps'], batch_size=batch_size, learning_rate=lr, early_stop_patience_steps=early_stop, val_check_steps=val_steps)
    
    if model_name == 'PatchTST':
        if model_conf['small']:
            return PatchTST(
                    input_size=input_size, h=model_conf['forecasting_horizon'], scaler_type='standard', 
                    patch_len=model_conf['input_size'], stride=24, revin=False, linear_hidden_size=64,hidden_size=16, n_heads=4, encoder_layers=3, 
                    max_steps=model_conf['max_steps'], batch_size=batch_size, learning_rate=lr, early_stop_patience_steps=early_stop, val_check_steps=val_steps)
        else:
            return PatchTST(
                    input_size=input_size, h=model_conf['forecasting_horizon'], scaler_type='standard', 
                    patch_len=model_conf['input_size'], stride=24, revin=False, linear_hidden_size=256,hidden_size=16, n_heads=4, encoder_layers=3, 
                    max_steps=model_conf['max_steps'], batch_size=batch_size, learning_rate=lr, early_stop_patience_steps=early_stop, val_check_steps=val_steps)
            
    if model_name == 'FEDformer':
        if model_conf['small']:
            return FEDformer(
                    input_size=input_size, h=model_conf['forecasting_horizon'], scaler_type='standard', 
                    modes=8, hidden_size=16, conv_hidden_size=16, n_head=8, windows_batch_size=32, 
                    max_steps=model_conf['max_steps'], batch_size=batch_size, learning_rate=lr, early_stop_patience_steps=early_stop, val_check_steps=val_steps)
        else:
            return FEDformer(
                    input_size=input_size, h=model_conf['forecasting_horizon'], scaler_type='standard', 
                    modes=32, hidden_size=64, conv_hidden_size=8, n_head=8, windows_batch_size=32, 
                    max_steps=model_conf['max_steps'], batch_size=batch_size, learning_rate=lr, early_stop_patience_steps=early_stop, val_check_steps=val_steps)
    