import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time

from dcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=1, help='Number of generated outputs')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

print(args.num_output)

# Display the generated image.
for i in range(2000):
	# Get latent vector Z from unit normal distribution.
	noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

	# Turn off gradient calculation to speed up the process.
	with torch.no_grad():
		# Get generated image from the noise vector using
		# the trained generator.
		generated_img = netG(noise).detach().cpu()
	
	plt.axis("off")
	plt.imshow(np.transpose(vutils.make_grid(generated_img, normalize=True, padding=0, pad_value=0), (1,2,0)))
	plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
				hspace = 0, wspace = 0)
	plt.margins(0,0)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.savefig("generated_images/"+ str(i) +".png", bbox_inches = 'tight',
		pad_inches = 0)
	# plt.show()