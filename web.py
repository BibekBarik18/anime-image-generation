#import the necessary libraries
import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

#Title 
st.title("Anime Image Generator")
st.divider()

#Get the number of images to be generated input from the user
num_images=st.slider("Pick the number of images to be generated",1,64)
noise_dim = 128  # Dimension of the noise vector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#denormalising the generated images
stats = (0.5,0.5,0.5),(0.5,0.5,0.5)
def denorm (img_tensors):
    return img_tensors*stats[1][0]+stats[0][0]

#Using matplotlib to plot the generated images
def show_images(images,nmax=64):
    images.cpu()
    fig,ax=plt.subplots(figsize=(8,8))
    ax.set_xticks([]);ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]),nrow=8).permute(1,2,0))

#Initialize the button to generate the image and display the image
if st.button("Generate Image"):
    generator=torch.load(r"D:\GIT\anime-image-generation\generator_model_asta1.pth")
    noise = torch.randn(num_images, noise_dim, 1, 1, device=device)
    with torch.no_grad():
        generated_images = generator(noise)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    img=show_images(generated_images.cpu())
    st.pyplot(img)