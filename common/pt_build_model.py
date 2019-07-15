import torch
import torch.nn.functional as F
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
    def __init__(self, old_model, input_dim, output_dim, state_dim=4):
        super(NonlinearTransformedModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = old_model
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.transform_model = torch.nn.Sequential(
              # torch.nn.Linear(input_dim, 128),
              # torch.nn.Linear(input_dim+state_dim, 128),
              torch.nn.Linear(input_dim+output_dim, 128),
              torch.nn.SELU(),
              torch.nn.AlphaDropout(.1),
              torch.nn.Linear(128, 128),
              torch.nn.SELU(),
              torch.nn.AlphaDropout(.1),
        )
        # pdb.set_trace()

        self.A_model = torch.nn.Linear(128, output_dim**2)
        # self.A_model = torch.nn.Linear(128, output_dim**2)
        self.D_model = torch.nn.Linear(128, output_dim)
        # self.D_model = torch.nn.Sequential(
        #         torch.nn.Linear(128, output_dim),
        #         torch.nn.Tanh(),
        #         torch.nn.Linear(output_dim, output_dim),
        #         )
        self.gate = torch.nn.Sequential(
                torch.nn.Linear(128, output_dim),
                torch.nn.Sigmoid()
                )

        self.rotation_matrix = np.diag(np.array([-1,-1,1,1]))
        self.iden = torch.autograd.Variable(torch.tensor(self.rotation_matrix, dtype = dtype))
            # Initialize transform as identity matrix
    def forward(self, inpt):
        out = self.model(inpt)#.detach()
        new_inpt = torch.cat((inpt, out), dim=-1)
        # pdb.set_trace()
        feats = self.transform_model(new_inpt)
        # feats = self.transform_model(inpt)
        # 
        D = self.D_model(feats)#*0.1
        alpha = self.gate(feats)
        # skip = False
        skip = True
        gradient_decay = .1

        if skip:
            D1 = D#*(1-gradient_decay)
            #D2 = (D*gradient_decay).detach()
            D_new = D1#+D2
        	# transformed_out = D
            # transformed_out = (1-alpha)*out + D*alpha
            pdb.set_trace()
            transformed_out = (1-alpha)*out + D_new*alpha
        else:
            A = self.A_model(feats)*.05
            if len(feats.shape) == 1:
            	A = A.view(self.output_dim, self.output_dim)
            elif len(feats.shape) == 2:
            	A = A.view(-1, self.output_dim, self.output_dim)
            else: 
            	print("Unhandled shape")
            	pdb.set_trace()

            mat = A + self.iden
            # out = self.model(inpt)#.detach()
            # transformed_out = (1-alpha)*out + D*alpha

            if len(feats.shape) == 1:
                # transformed_out = torch.matmul(mat, out) + D
                t_out = torch.matmul(mat, out)
            elif len(feats.shape) == 2:
                out2 = torch.unsqueeze(out, -1)
                # transformed_out = (torch.bmm(mat,out2)*.0 + D.unsqueeze(-1)).squeeze(-1)
                t_out = torch.bmm(mat,out2).squeeze(-1)

            transformed_out = (1-alpha)*t_out + D*alpha
            # transformed_out = out + D
        # pdb.set_trace()
        return transformed_out

    def set_base_model_train(self, setting):
        for param in self.model.parameters():
            param.requires_grad = setting


# class RecurrentNet(torch.nn.Module):
#     def __init__(input_dim, output_dim, dropout_p=.1):
#         self.l1 = torch.nn.Linear(input_dim, 128)
#         self.l2 = torch.nn.LSTM(128, 128, batch_first=True, dropout=dropout_p)




def pt_build_model(nn_type, input_dim, output_dim, dropout_p=.1):
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
    # elif nn_type == 'LSTM':
    #     model = torch.nn.LSTM(input_dim, output_dim, num_layers=3, dropout=dropout_p)


	# pdb.set_trace()
	return model


class BNNWrapper(torch.nn.Module):
    def __init__(self, model, input_dim, output_dim):
        super(BNNWrapper, self).__init__()
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim

    def get_distro(self,x):
        # pdb.set_trace()
        output = self.model(x)
        means = output[...,: self.output_dim]
        stds = F.elu(output[..., self.output_dim:]) + 1
            #Make sure Standard Deviation > 0
        # pdb.set_trace()
        distro = torch.distributions.normal.Normal(means, stds)
        return distro

    def forward(self,x, true_state=None):
        distro = self.get_distro(x)
        # sample = distro.sample()
        sample = distro.mean
        # interp = .9
        # sample = distro.mean*interp+ distro.sample()*(1-interp)
        if true_state is not None: 
            log_p = distro.log_prob(true_state)
            nan_locs = (log_p != log_p) #Get locations where log_p is undefined
            if nan_locs.any():
                pdb.set_trace()
            log_p[nan_locs] = 0 #Set the loss in those locations to 0
            return sample, log_p

        return sample
        # return distro.mean








