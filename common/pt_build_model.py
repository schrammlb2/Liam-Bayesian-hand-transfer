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

class NonlinearTransformedModel(torch.nn.Module):
    def __init__(self, old_model, input_dim, output_dim):
        super(NonlinearTransformedModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = old_model
        for param in self.model.parameters():
            param.requires_grad = False

        self.transform_model = torch.nn.Sequential(
              torch.nn.Linear(input_dim, 128),
              torch.nn.SELU(),
              torch.nn.AlphaDropout(.1),
              torch.nn.Linear(128, 128),
              torch.nn.SELU(),
              torch.nn.AlphaDropout(.1),
        )

        self.A_model = torch.nn.Linear(128, output_dim**2)
        # self.A_model = torch.nn.Linear(128, output_dim**2)
        self.D_model = torch.nn.Linear(128, output_dim)

        self.rotation_matrix = np.diag(np.array([-1,-1,1,1]))
        self.iden = torch.autograd.Variable(torch.tensor(self.rotation_matrix, dtype = dtype))
            # Initialize transform as identity matrix
    def forward(self, inpt):
        # return self.lt(self.model(self.lt_inv(inpt)))

        # pdb.set_trace()
        feats = self.transform_model(inpt)

        A = self.A_model(feats)*.015
        D = self.D_model(feats)*.1
        skip = True

        if skip:
        	transformed_out = D
        else:
            if len(feats.shape) == 1:
            	A = A.view(self.output_dim, self.output_dim)
            elif len(feats.shape) == 2:
            	A = A.view(-1, self.output_dim, self.output_dim)
            else: 
            	print("Unhandled shape")
            	pdb.set_trace()

            mat = A + self.iden
            out = self.model(inpt).detach()

            if len(feats.shape) == 1:
                transformed_out = torch.matmul(mat, out) + D
            elif len(feats.shape) == 2:
                out2 = torch.unsqueeze(out, -1)
                transformed_out = (torch.bmm(mat,out2) + D.unsqueeze(-1)).squeeze(-1)
            # transformed_out = out + D
        # pdb.set_trace()
        return transformed_out



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