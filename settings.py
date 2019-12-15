def get_cuda():
	cuda = torch.cuda.is_available()
	cuda = False
	return cuda