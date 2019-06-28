import torch
import pdb
import numpy as np

dtype = torch.float



class LinearTransformedModel(torch.nn.Module):
    def __init__(self, old_model, input_dim, output_dim):
        super(LinearTransformedModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = old_model
        for param in self.model.parameters():
            param.requires_grad = False

        # self.lt = torch.nn.Linear(input_dim, input_dim)
        # self.lt_inv = torch.nn.Linear(output_dim, output_dim)

        self.mat = torch.autograd.Variable(torch.tensor(np.identity(input_dim), dtype = dtype), requires_grad=True)
        self.mat_inv = self.mat[:output_dim, :output_dim].transpose(0,1)
            # Initialize transform as identity matrix
    def forward(self, inpt):
        # return self.lt(self.model(self.lt_inv(inpt)))

        # pdb.set_trace()

        trans_in = torch.matmul(inpt,self.mat)
        out = self.model(trans_in)
        detrans_out = torch.matmul(out,self.mat_inv)

        return detrans_out

    def get_consistency_loss(self, inpt):
        # trans_state = self.lt(inpt)[:self.output_dim]
        # detrans_state = self.lt_inv(trans_state)
        # mse_fn = torch.nn.MSELoss()
        # return mse_fn(detrans_state, inpt[:self.output_dim])


        trans_state = torch.mm(self.mat, inpt)[:self.output_dim]
        detrans_state = torch.mm(self.mat_inv, trans_state)
        mse_fn = torch.nn.MSELoss()
        return mse_fn(detrans_state, inpt[:self.output_dim])


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
		      torch.nn.SELU(),
		      torch.nn.AlphaDropout(dropout_p),
		      torch.nn.Linear(128, 128),
		      torch.nn.SELU(),
		      torch.nn.AlphaDropout(dropout_p),
		      torch.nn.Linear(128, output_dim),
		)
	elif nn_type == '2':
		model = torch.nn.Sequential(
		      torch.nn.Linear(input_dim, 128),
		      torch.nn.Tanh(),
		      torch.nn.Dropout(dropout_p),
		      torch.nn.Linear(128, 64),
		      torch.nn.SELU(),
		      torch.nn.AlphaDropout(dropout_p),
		      torch.nn.Linear(64, 32),
		      torch.nn.SELU(),
		      torch.nn.AlphaDropout(dropout_p),
		      torch.nn.Linear(32, output_dim),
		)


	# pdb.set_trace()
	return model