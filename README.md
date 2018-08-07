Tools to handle radio-interferometric observations.
By Enrique Macias

Useful functions:
- get_vis_obs: to be run within CASA, it retrieves the visibilities from a
CASA measurement set and saves them into different arrays in a vis_obj object.
- get_sim_model: to be run within CASA, it simulates an observation from a
synthetic model image using the uv-coverage of an observed measurement set.
It saves the model and residuals in new measurement sets and in vis_obj objects.
- export_fits: save the visibilities in a fits file that can be easily imported.
- deproject: method of the vis_obj object, it deprojects the visibilities.
- phase_shift: method of the vis_obj object, it applies a shift to the phase
center of the visibilities.
- bin_vis: method of the vis_obj object, it bins the visibilities.

Example:
(Within CASA)
vis = get_vis_obs('myms.ms')
vis.export_fits(outfile='myvis') # Saves the visibilities in myvis.fits

(Within CASA or python)
vis = vis_tools.vis_obj(input_file='myvis.fits')
vis.deproject(45.0,45.0) # deproject visibilities
vis.phase_shift(10.0,-10.0) # apply a shift to phase center
vis.plot_vis(binned=False,outfile='plot_deproj_real') # Makes plot of deprojected visibilities
vis.bin_vis(nbins=100,deproj=True) # Bin visibilities in 100 bins
vis.plot_vis(outfile='plot_deproj_binned_real') # Makes plot of binned visibilities
