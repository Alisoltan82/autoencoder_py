#%%
import numpy as np
import pandas as pd
import torch
from torch import nn
import monai
import torchvision
import matplotlib.pyplot as plt
# %%
from monai.networks.nets import AutoEncoder
from monai.transforms import Compose , DivisiblePad , ToTensor
from torchvision.models.feature_extraction import get_graph_node_names , create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import cv2
import os
from glob import glob
# %%
image = cv2.imread(r'E:\autoencoder/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis('off')
# %%
image = cv2.resize(image,(0,0) , fx = 0.5 , fy = 0.5)/255
image.shape

#%%
img_list = []
img_list.append(image)
img_list = np.asarray(img_list)
img_list.shape

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoEncoder(
    spatial_dims = 2,
    in_channels = 3,
    out_channels = 3,
    channels = (64,128 , 256 , 512),
    strides = (2,2,2,2)
).to(device)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters() , lr = 1e-4)

trans = Compose([
    ToTensor()
])
img_list_t = img_list.transpose((0,3,1,2))
img_array = trans(img_list_t.astype(np.float32))
img_array.shape , img_array.dtype
# %%
#padding per image for the 3 channels then stack back
pad_trans = Compose([
    DivisiblePad(k = 64 , mode = 'symmetric')
])
list_ = []
for i in range(3):
  img = img_list_t[:,i,:,:]
  img = pad_trans(img)
  print(img.shape)
  list_.append(img)
list_ = np.asarray(list_,dtype = np.float32)
list_.shape
# %%
list_ = list_.transpose((1,0,2,3))
print(list_.shape , list_.dtype)
# %%
tensor_array = trans(list_)
tensor_array.shape
# %%
#training loop
max_epochs = 100
epoch_loss_values = []

for epoch in range(max_epochs):
  model.train()
  epoch_loss = 0
  step = 0
  input = tensor_array.to(device)

  step+=1
  optimizer.zero_grad()
  output = model(input)
  loss = loss_function(output , input)
  loss.backward()

  optimizer.step()
  epoch_loss += loss.item()

  epoch_loss /= step
  epoch_loss_values.append(epoch_loss)
  print(f'EPOCH:{epoch} **** epochloss:{epoch_loss:.4f}')
#%%
#extract layers names
train_nodes,eval_nodes = get_graph_node_names(model)
train_nodes
# %%
#define the layers to extract
return_nodes = ['encode.encode_1.conv' , 'encode.encode_2.conv']
feat_ex = create_feature_extractor(model , return_nodes = return_nodes)
# %%
with torch.inference_mode():
  out = feat_ex(tensor_array)
  print(out.keys())

# %%
print(out['encode.encode_1.conv'].shape)
print(out['encode.encode_2.conv'].shape)
# %%
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(10, 10), tight_layout=True)
import random
# flatten the axis into a 1-d array to make it easier to access each axes
axes = axes.flatten()

# iterate through axes and associated file
files = out['encode.encode_1.conv'].numpy().squeeze(0)
for ax, file in zip(axes, files):
    for i in range(20):
       rand = random.choice(files[i:,:,:])
    
       # add the image to the axes
       ax.imshow(rand , cmap = 'bone')

    # remove ticks / labels
    ax.axis('off')


# %%
