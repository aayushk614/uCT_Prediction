import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import h5py

import imageio
import h5py
import os
import imageio

import torch
import os,datetime
import torch.nn as nn
from torchsummary import summary

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

st.title('Volumetric Segmentation of Drosophila Longitudinal muscles')

def read_h5(filename, dataset=''):
    fid = h5py.File(filename, 'r')
    if dataset == '':
        dataset = list(fid)[0]
    return np.array(fid[dataset])


def read_h5(filename, dataset=''):
    fid = h5py.File(filename, 'r')
    if dataset == '':
        dataset = list(fid)[0]
    return np.array(fid[dataset])


# Add in location to select image.

st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['h5'],
                                         accept_multiple_files=False)

col1, col2, col3 = st.beta_columns(3)


if st.sidebar.button('Click to Analyze'):
    

    # User-selected Volume.

    volume = read_h5(uploaded_file).astype(np.uint8)

    # Display image.
    st.write('Shape of Input Volume :')
    st.write(volume.shape)
    st.write('Data Type of Input Volume :')
    st.write(volume.dtype)

    
    for i in range(0,112,2):
        slice = volume[i,:,:]

        st.write('Slice no. :', i)    
        st.image(slice)


    # load pre-trained model
    checkpoint = 'checkpoint_95000.pth.tar'

    from threedunet import unet_residual_3d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model= unet_residual_3d(in_channel=1, out_channel=13).to(device)
    model = nn.DataParallel(model, device_ids=range(1))
    model = model.to(device)

    summary(model,(1,112,112,112))

    st.write('Load pretrained checkpoint: ', checkpoint)
    checkpoint = torch.load(checkpoint)
    st.write('checkpoints: ', checkpoint.keys())

    # update model weights
    if 'state_dict' in checkpoint.keys():
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.module.state_dict() # nn.DataParallel
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict 
        model_dict.update(pretrained_dict)    
        # 3. load the new state dict
        model.module.load_state_dict(model_dict) # nn.DataParallel
        
        st.write("new state dict loaded ")


    
    volume = torch.from_numpy(vol).to(device, dtype=torch.float)
    volume = volume.unsqueeze(0)

    volume = volume.unsqueeze(0)

    st.write('Model Input shape',volume.shape)

    pred = model(volume)
    st.write("Shape of pred after test", pred.shape)
    pred = pred.squeeze(0)
    st.write("Shape of pred after test", pred.shape)
    pred = pred.cpu()
    arr1 = np.argmax(pred.detach().numpy(),axis=0).astype(np.uint16)
    st.write("shape of Predictions after argmax() function ", arr1.shape)

    
    hf1 = h5py.File('pred_generated.h5', 'w')
    hf1.create_dataset('dataset1', data=arr1)
    st.write("Predicted volume created and saved" , hf1)
    hf1.close()


    st.write('SUCCESS!!!!!!!!')
    


