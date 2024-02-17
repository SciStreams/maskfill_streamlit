import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np 

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from astropy.io import fits 
from maskfill import maskfill 


@st.cache_data
def import_m51_sample():

    image_clean = fits.getdata('./example_m51/m51_org.fits')
    image_cr = fits.getdata('./example_m51/m51_with_cosmicrays.fits')
    mask = fits.getdata('./example_m51/m51_mask.fits')

    return image_clean, image_cr, mask


@st.cache_data
def run_mfill(image_clean, image_cr, mask):

    mfill_smooth, mfill = maskfill(image_cr, mask)

    return mfill_smooth, mfill


def show_data(image_clean, image_cr, mask):


    # Create subplots
    fig = make_subplots(rows=1, cols=6,
                        subplot_titles=("Original", "Cosmic Rays", "Mask"),
                        specs=[[{}, None, {}, None, {}, None]],
                        shared_yaxes=True)

    # Range
    vmin,vmax=np.percentile(image_clean,[1,99])


    fig.add_trace(go.Heatmap(
        z=image_clean,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.26, title='', thickness=15)), row=1, col=1
    )

    fig.add_trace(go.Heatmap(
        z=image_cr,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.61, title='', thickness=15)), row=1, col=3
    )


    fig.add_trace(go.Heatmap(
        z=mask,
        colorscale='gray_r',
        zmin=0.1,
        zmax=0.9,
        colorbar=dict(x=0.96, title='', thickness=15)), row=1, col=5
    )

    # Update layout
    fig.update_layout(
        title_text='Example M51',
        title_x=0.4,
        title_y=1,
        height=250,
        margin=dict(l=0, r=0, b=40, t=45),
        xaxis=dict(domain=[0, 0.27]),
        xaxis2=dict(domain=[0.35, 0.62]),
        xaxis3=dict(domain=[0.7, 0.97])
    )

    fig.update_xaxes(matches='x')

    st.plotly_chart(fig)


def simple_run(image_clean, image_cr, mask, mfill_smooth, mfill):


    # Create subplots
    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=("maskfill smooth", "maskfill"),
                        specs=[[{}, None, {}, None]],
                        shared_yaxes=True, shared_xaxes=True)

    # Range
    vmin,vmax=np.percentile(image_clean,[1,99])


    trace1 = go.Heatmap(
        z=mfill_smooth,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.44, title='', thickness=15)
    )
    fig.add_trace(trace1, row=1, col=1)

    trace2 = go.Heatmap(
        z=mfill,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.96, title='', thickness=15)
    )
    fig.add_trace(trace2, row=1, col=3)

    # Update layout
    fig.update_layout(
        title_text='Output from simple maskfill run',
        title_x=0.4,
        title_y=1,
        height=350,
        margin=dict(l=0, r=0, b=40, t=45),
        xaxis=dict(domain=[0, 0.45]),
        xaxis2=dict(domain=[0.55, 0.97])
    )

    fig.update_xaxes(matches='x')

    # Display the figure
    st.plotly_chart(fig)


def compare_image_simple_run(image_clean, image_cr, mask, mfill_smooth, mfill, vmin, vmax):

    st.write("You can zoom into the region of image with the mouse selection.")

        # Create subplots
    fig = make_subplots(rows=1, cols=6,
                        subplot_titles=("Original", "maskfill", "maskfill smooth"),
                        specs=[[{}, None, {}, None, {}, None]],
                        shared_yaxes=True)


    fig.add_trace(go.Heatmap(
        z=image_clean,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.26, title='', thickness=15)), row=1, col=1
    )

    fig.add_trace(go.Heatmap(
        z=mfill,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.61, title='', thickness=15)), row=1, col=3
    )


    fig.add_trace(go.Heatmap(
        z=mfill_smooth,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.96, title='', thickness=15)), row=1, col=5
    )

    # Update layout
    fig.update_layout(
        title_text='Result From M51 Example',
        title_x=0.4,
        title_y=1,
        height=250,
        margin=dict(l=0, r=0, b=40, t=45),
        xaxis=dict(domain=[0, 0.27]),
        xaxis2=dict(domain=[0.35, 0.62]),
        xaxis3=dict(domain=[0.7, 0.97])
    )

    fig.update_xaxes(matches='x')
    fig.update_layout({"uirevision": "foo"}, overwrite=True)
    st.plotly_chart(fig)


def indices_within_distance(array, x, y, distance):
    """
    Creating an (arbitrary) large mask on a cutout from the m51 image.
    https://maskfill.readthedocs.io/en/latest/python-usage.html#intermediate-outputs
    """
    rows, cols = np.indices(array.shape)
    distances = np.sqrt((rows - x)**2 + (cols - y)**2)
    indices = np.where(distances <= distance)
    return indices


@st.cache_data
def create_cutout(image_clean):

    image_cutout  = image_clean[390:490,400:500]
    cutout_mask = np.zeros(image_cutout.shape)
    cutout_mask[indices_within_distance(image_cutout,50,50,48)] = 1

    # Convert the masked array to a regular NumPy array
    masked_image = np.ma.masked_array(image_cutout, mask=cutout_mask.astype(bool)).filled(fill_value=np.nan)

    return image_cutout, masked_image


def see_image_cutout(image_cutout, masked_image):

    #image_cutout  = image_clean[390:490,400:500]
    #cutout_mask = np.zeros(image_cutout.shape)
    #cutout_mask[indices_within_distance(image_cutout,50,50,48)] = 1

    vmin,vmax=np.percentile(image_cutout,[1,99])




    st.write("Examples of creating an (arbitrary) large mask on a cutout from the M51 image")

        # Create subplots
    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=("Cutout", "Mask"),
                        specs=[[{}, None, {}, None]],
                        shared_yaxes=True, shared_xaxes=True)


    trace1 = go.Heatmap(
        z=image_cutout,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.44, title='Cutout', thickness=15)
    )
    fig.add_trace(trace1, row=1, col=1)

    trace2 = go.Heatmap(
        z=masked_image,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.96, title='Cutout', thickness=15)
    )
    fig.add_trace(trace2, row=1, col=3)

    # Update layout
    fig.update_layout(
        title_text='Masking Region',
        title_x=0.4,
        title_y=1,
        height=350,
        margin=dict(l=0, r=0, b=40, t=45),
        xaxis=dict(domain=[0, 0.45]),
        xaxis2=dict(domain=[0.55, 0.97])
    )

    fig.update_xaxes(matches='x')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout({"uirevision": "foo"}, overwrite=True)
    # Display the figure
    st.plotly_chart(fig)


def one_kernel_size(image_clean, cutout_mask, fill_reg):

    image_cutout  = image_clean[390:490,400:500]
    cutout_mask = np.zeros(image_cutout.shape)
    cutout_mask[indices_within_distance(image_cutout,50,50,48)] = 1
    vmin,vmax=np.percentile(image_cutout,[1,99])

    fill1, _ = maskfill(image_cutout,cutout_mask,size=fill_reg)


        # Create subplots
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=(""),
                        shared_yaxes=True)


    fig.add_trace(go.Heatmap(
        z=fill1,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.98, title='', thickness=15)), row=1, col=1
    )


    # Update layout
    fig.update_layout(
        title_text=f'Kernel Size {fill_reg}',
        title_x=0.3,
        title_y=1,
        height=300,
        width=300,
        margin=dict(l=0, r=0, b=40, t=45),
        xaxis=dict(domain=[0, 0.97]),
    )

    fig.update_xaxes(matches='x')
    fig.update_layout({"uirevision": "foo"}, overwrite=True)
    st.plotly_chart(fig)


def kernel_size(image_clean, cutout_mask):

    image_cutout  = image_clean[390:490,400:500]
    cutout_mask = np.zeros(image_cutout.shape)
    cutout_mask[indices_within_distance(image_cutout,50,50,48)] = 1
    vmin,vmax=np.percentile(image_cutout,[1,99])

    fill1, _ = maskfill(image_cutout,cutout_mask,size=3)
    fill2, _ = maskfill(image_cutout,cutout_mask,size=5)
    fill3, _ = maskfill(image_cutout,cutout_mask,size=11)


        # Create subplots
    fig = make_subplots(rows=1, cols=6,
                        subplot_titles=("Kernel 3", "Kernel 5", "Kernel 11"),
                        specs=[[{}, None, {}, None, {}, None]],
                        shared_yaxes=True)


    fig.add_trace(go.Heatmap(
        z=fill1,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.26, title='', thickness=15)), row=1, col=1
    )

    fig.add_trace(go.Heatmap(
        z=fill2,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.61, title='', thickness=15)), row=1, col=3
    )


    fig.add_trace(go.Heatmap(
        z=fill3,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(x=0.96, title='', thickness=15)), row=1, col=5
    )

    # Update layout
    fig.update_layout(
        title_text='Kernel Size Examples for selected region in M51',
        title_x=0.3,
        title_y=1,
        height=250,
        margin=dict(l=0, r=0, b=40, t=45),
        xaxis=dict(domain=[0, 0.27]),
        xaxis2=dict(domain=[0.35, 0.62]),
        xaxis3=dict(domain=[0.7, 0.97])
    )

    fig.update_xaxes(matches='x')
    fig.update_layout({"uirevision": "foo"}, overwrite=True)
    st.plotly_chart(fig)