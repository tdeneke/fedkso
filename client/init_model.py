from fedn.utils.pytorchhelper import PytorchHelper
from models.mnist_model import create_seed_model

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = create_seed_model()
	outfile_name = "initial_model.npz"

	weights = model.state_dict()

	helper = PytorchHelper()
	helper.save_model(weights, outfile_name)