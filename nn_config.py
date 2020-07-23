
############################
# PARAMETER CONFIGURATIONS #
############################


def set_data_params(num_train):
    params = {}
    params['batch_size'] = 100
    params['n_iters'] = 6000
    params['num_epochs'] = int(params['n_iters'] / num_train /params['batch_size'])
    return params


def set_network_params():
    params = {}
    params['input_dim'] = 30
    params['hidden_dim'] = 128
    params['n_layer'] = 2
    params['output_dim'] = 10
    params['learning_rate'] = 0.1
    params['sequence_dim'] = 28
    return params