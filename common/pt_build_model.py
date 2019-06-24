import torch
import pdb
def pt_build_model(nn_type, input_dim, output_dim, dropout_p):
	if nn_type == '0':
		model = torch.nn.Sequential(
		      torch.nn.Linear(input_dim, 128),
		      torch.nn.ReLU(),
		      torch.nn.Dropout(dropout_p),
		      torch.nn.Linear(128, 128),
		      torch.nn.ReLU(),
		      torch.nn.Dropout(dropout_p),
		      torch.nn.Linear(128, output_dim),
		)
	elif nn_type == '1':
		model = torch.nn.Sequential(
		      torch.nn.Linear(input_dim, 128),
		      torch.nn.SeLU(),
		      torch.nn.AlphaDropout(dropout_p),
		      torch.nn.Linear(128, 128),
		      torch.nn.SeLU(),
		      torch.nn.AlphaDropout(dropout_p),
		      torch.nn.Linear(128, output_dim),
		)
	# pdb.set_trace()
	return model