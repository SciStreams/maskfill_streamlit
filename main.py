import streamlit as st
from maskfill import maskfill

import numpy as np
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots

from helper_functions import import_m51_sample, run_mfill, show_data, simple_run, compare_image_simple_run,\
								see_image_cutout, create_cutout, indices_within_distance, kernel_size,\
								one_kernel_size



image_clean, image_cr, mask = import_m51_sample()
mfill_smooth, mfill = run_mfill(image_clean, image_cr, mask)

# Range
vmin_im, vmax_im = np.percentile(image_clean,[1,99])

# Cast vmin_im and vmax_im to float
vmin_im = float(vmin_im)
vmax_im = float(vmax_im)


with st.sidebar:
	sel_options = st.radio("Select Options", ['See Sample', 'Compare Results [Simple Run]', 'Optional Arguments', 'References'])



if sel_options == "See Sample":
	st.header("Imported Sample")

	st.write("Loading the uncorrupted and corrupted versions of the M51 example presented in [van Dokkum & Pasha (2024)](https://browse.arxiv.org/abs/2312.03064):")


	show_data(image_clean, image_cr, mask)


	st.subheader("Simple maskfill run:")
	st.write("Results from an option-free run:")
	simple_run(image_clean, image_cr, mask, mfill_smooth, mfill)



if sel_options == "Compare Results [Simple Run]":
	#compare_image_simple_run(image_clean, image_cr, mask, mfill_smooth, mfill, vmin_im, vmax_im)

	#import plotly.figure_factory as ff


	if "vmin" not in st.session_state:
	    st.session_state["vmin"] = vmin_im
	if "vmax" not in st.session_state:
	    st.session_state["vmax"] = vmax_im

	# put figure inside a form?
	with st.form("form2"):
		st.write("Click on the Button below to compare results on the plot. You can enhance fainter regions by selecting a smaller maximum value of 'Select range' and then click on the Button again to update plot.")
		submitted2 = st.form_submit_button(label="Show Plot and Update")
		vals = st.slider("Select range", vmin_im, vmax_im, (vmin_im, vmax_im))

		new_vmin = vals[0] #st.slider("Min", min_value=vmin_im, max_value=vmax_im, value=vmin_im, step=0.1)
		new_vmax = vals[1] #st.slider("Max", min_value=vmin_im, max_value=vmax_im, value=vmax_im, step=0.1)

	if submitted2:
		#mfill_smooth, mfill = maskfill(image_cr, mask)
		
		compare_image_simple_run(image_clean, image_cr, mask, mfill_smooth, mfill, new_vmin, new_vmax)

if sel_options == "Optional Arguments":
	with st.sidebar:
		arg_options = st.radio("Arguments", ["Window Size"])

		
	if arg_options == "Window Size":

		st.header("See impact of window size selection")
		st.write('" This argument controls the window of adjacent pixels used for computation of the local medians, and for the final smoothing step" [van Dokkum & Pasha (2024)](https://maskfill.readthedocs.io/en/latest/python-usage.html#intermediate-outputs)')

		image_cutout, masked_image = create_cutout(image_clean)
		see_image_cutout(image_cutout, masked_image)

		
		ker_option = st.radio("Choose option", ["Single Kernel Size", "Multiple Kernels Comparison"])

		if ker_option == "Single Kernel Size":

			fill_reg = st.slider("Kernel Size", min_value=3, max_value=15, value=3, step=2)
			one_kernel_size(image_clean, masked_image, fill_reg)

		if ker_option == "Multiple Kernels Comparison":
			
		
			kernel_size(image_clean, masked_image)

if sel_options == "References":
	st.header("References")
	st.write(
		"""Python package Maskfill is made by Pieter van Dokkum & Imad Pasha, see [van Dokkum & Pasha (2024)](https://browse.arxiv.org/abs/2312.03064)
	""")
	st.write("")
	st.write(""" 
		Check Maskfill [**documentation**](https://maskfill.readthedocs.io/en/latest/index.html).
		""")
	st.write("")
	st.write(""" 
		Maskfill code is available on [**GitHub**](https://github.com/dokkum/maskfill).
		""")

	st.write(""" 
		This Streamlit application is made by SciStreams, code of the app can be found on [**GitHub**](https://github.com/SciStreams/maskfill_streamlit).
		""")
	st.write("")
	st.write(""" 
	If you find maskfill useful in your research or image handlings, please cite the code:

	@ARTICLE{2023arXiv231203064V,
       author = {{van Dokkum}, Pieter and {Pasha}, Imad},
        title = "{A robust and simple method for filling in masked data in astronomical images}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = dec,
          eid = {arXiv:2312.03064},
        pages = {arXiv:2312.03064},
          doi = {10.48550/arXiv.2312.03064},
	archivePrefix = {arXiv},
       eprint = {2312.03064},
	 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv231203064V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
	}
			""")


with st.sidebar:
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	
	st.sidebar.info(
		"""This Streamlit app showcase examples from **maskfill** Python package.
		Original [**maskfill**](https://maskfill.readthedocs.io/en/latest/index.html) Python package 
		is made by Pieter van Dokkum & Imad Pasha, see [van Dokkum & Pasha (2024)](https://browse.arxiv.org/abs/2312.03064)
		"""
		)
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	st.write("")
	st.write("")

	col1, col2 = st.columns([0.7,0.2])
	with col1:

		st.markdown('''
	    <a href="https://scistreams.github.io">
	        <img src="https://scistreams.github.io/images/SciStreams.png" width="150" />
	    </a>''',
	    unsafe_allow_html=True
		)
		st.markdown('App made by [**SciStreams**](https://scistreams.github.io/)')