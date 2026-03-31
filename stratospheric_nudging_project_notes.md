# Notes on ACE2 stratospheric nudging project

These notes are intended as a guide for Myles and a potential RSE to set out the overall motivation and some milestones for the ACE2 stratospheric nudging project. 

## Scientific background and motivation

[ACE2](https://github.com/ai2cm/ace) is a ML-based emulator for weather and climate. It has been trained on various datasets from observations and physics-based models. In this project we will use [ACE2-ERA5](https://huggingface.co/allenai/ACE2-ERA5), which is trained on the ERA5 reanalysis (a gridded observational dataset). ACE2 has been shown to perform well compared to physics-based models from subseasonal to decadal scales (see [Watt Meyer et al. 2025](https://www.nature.com/articles/s41612-025-01090-0). 

A known issue with many ML-based weather and climate models is poor performance in the stratosphere. Stratospheric variability is known to influence surface weather, particularly during winter when rapid breakdowns of the stratospheric polar vortex, known as sudden stratospheric warmings (SSWs), can drive extreme cold air outbreaks (see, e.g.,[Domeisen and Butler, 2020](https://www.nature.com/articles/s43247-020-00060-z)). 

In this project we will examine the extent to which a 'perfect' stratosphere might improve ACE2 predictions by using a technique called 'nudging'. Nudging simply forces a chosen region of the atmosphere to follow a prescribed evolution. Typically, in physics-based models, this is done with a given relaxation time scale, but we will force ACE2 to exactly follow the trajectory (essentially setting the time scale to 0). The region we will nudge will be the uppermost level of ACE2-ERA5, representing approximately 50 hPa to the top of the atmosphere. We will nudge only the zonal wind at this level - called `eastward_wind_0` in ACE2.

A key aim is that our nudged ML predictions are directly comparable to a set of simulations in physics-based models within the [Stratospheric Nudging and Predictable Surface Impacts (SNAPSI) project](https://aparc-climate.org/data-centre/snapsi/). These cover 3 case-study SSW events, with forecasts run for about 3 months, and include free running experiments and experiments with the stratosphere nudged to ERA5 and to a climatological state. Our ACE2 experiments will be most comparable to the *nudged-full* SNAPSI experiments since these nudge at each gridpoint separately rather than to a zonal-mean.  

Finally, the broader goal is that by demonstrating that nudging can be implemented in a ML-based model and evaluating the results, we can motivate a wider comparison of nudged ML models (maybe called SNAPSI-AI!).


## Setting up ACE2 with stratospheric 'nudging'

Below I will give a brief outline for setting up ACE2-ERA5 with stratospheric nudging.

1. We have been using the `maths-gpu` server at Exeter. 
1. Clone ACE2-ERA5 from https://huggingface.co/allenai/ACE2-ERA5 (you'll need something like `git-lfs` for this). 
1. Fork and clone the ACE2 repo here: https://github.com/ai2cm/ace
1. The `stepper_override` feature is capable of doing the nudging, but seems not to work on the current main branch of ACE2. Therefore we need to checkout this commit: [170e1b6194c6d9539e7be557dc421cd03ba1703c](https://github.com/ai2cm/ace/commit/170e1b6194c6d9539e7be557dc421cd03ba1703c). You can also just use the branch that this guide is in!. Install ace with `pip install -e`.
1. We next need to modify a forcing file so that it includes the `eastward_wind_0` variable from ERA5. An ERA5 dataset in the correct ACE2 format is available at `/disco/share/ws359/ERA5_for_ACE` and the standard forcing files (which contain variables like sea-surface temperature) are in the ACE2-ERA5 repo (in `forcing_data`). Assuming we are running for the year 2018, we'll need to take `eastward_wind_0` from 2018 in the ERA5 data and add it to the file `forcing_2018.nc`. I'd recommend that the new modified forcing data file be placed in a new directory (e.g. `mod_forcing_data`). 
1. Next we need an initial condition file for ACE. A file for 2018-01-25 (the first initialisation in SNAPSI) is in: `/disco/share/ws359/ERA5_initial_conditions_for_ACE`
1. We then need to modify the YAML file that controls ACE. Below is an example:
```yaml
experiment_dir: ./output_directory
n_forward_steps: 400 
forward_steps_in_memory: 50
checkpoint_path: ./ace2_era5_ckpt.tar
logging:
  log_to_screen: true
  log_to_wandb: false
  log_to_file: true
  project: ace
initial_condition:
  path: /disco/share/ws359/ERA5_initial_conditions_for_ACE/era5_ic_20180125T00_20180326T23_05deg.nc
  start_indices:
    times:
      - "2018-01-25T00:00:00"
forcing_loader:
  dataset:
    data_path: ./mod_forcing_data
  num_data_workers: 4
stepper_override:
  prescribed_prognostic_names: ['eastward_wind_0']
data_writer:
  save_prediction_files: true
  save_monthly_files: false
  names: ['TMP2m', 'VGRD10m', 'PRATEsfc','air_temperature_0', 'air_temperature_1', 'air_temperature_2' , 'air_temperature_3', 'air_temperature_4', 'air_temperature_5', 'air_temperature_6', 'air_temperature_7', 'eastward_wind_0', 'eastward_wind_1']
```
1. Finally we should be able to run the inference with `python -m fme.ace.inference name_of_config_file.yaml`


## Science plan

Here I'll give a brief outline of the science milestones I'm hoping we can reach. 

### Main goals
1. First we should do a sanity-check that the nudging is working and affecting the atmosphere at other levels (I'm pretty sure this is the case, but it makes sense to check first). So run with and without nudging from the same initial condition to check this. 
1. Next we should run ensembles (ideally of size 50, to match SNAPSI, though maybe start with a smaller number) of free-running and nudged predictions. Start with the two initialisation dates of the 2018 case study. We can produce ensembles by adding gridpoint noise (Myles has done some work to understand an appropriate amount of noise to add). It would be worth writing some code to automate this as much as possible. 
1. Compare some key variables for these experiments. For instance, look at zonal-mean zonal wind at level 0 (ERA5 and ACE2 nudged should match, while the free running case will not). Look at surface temperature, comparing the ensemble mean difference between nudged and free to the anomaly in ERA5. It would be a good idea to calculate statistical significance for the surface temperature difference. Do likewise for sea-level pressure, and maybe the NAO/AO which can be derived from it. 
1. For further comparison with SNAPSI we can also run a *control* experiment by making a forcing file that has climatological mean `eastward_wind_0`. Some care will be needed to take account of leap days in doing this. 
1. The results here will be most comparable with the SNAPSI WG1 paper (led by Hera Kim, to be submitted soon). However, it would be nice to also make some of your own plots from SNAPSI data for direct comparison. SNAPSI data is available on JASMIN at `/badc/snap/data/post-cmip6/SNAPSI`, and there is a group workspace with some processed data at `/gws/nopw/j04/snapsi/processed`. The key question I'd like to answer is how much stratospheric nudging impacts ensemble mean prediction in ACE compared to SNAPSI.


### Additional 'would be nice' things
1. It would be nice to get an overall idea for the mean strength of stratosphere-troposphere coupling in ACE2, not just the particular SNAPSI case studies. Oli Watt-Meyer (who leads ACE) has also said he'd be interested in this result. To do this, we'd need to run a long 'climate' run of ACE2 (or likely an ensemble of these). Things to look at would be lead-lag correlations between stratospheric polar vortex strength and the surface (e.g. AO and NAO) - this might be compared to e.g. [Perlwitz and Harnik 2004](https://doi.org/10.1175/JCLI-3247.1) as well as our own calculations. 
1. Following on from the previous point, it would be great to construct something like the famous 'dripping paint' plots [Baldwin and Dunkerton 2001](https://www.science.org/doi/10.1126/science.1063315) of the NAM composited around SSWs. However, I *think* ACE2-ERA5 only has geopotential height at 500hPa. We'd therefore need to think about how to calculate the geopotential height from other variables.  
1. It would be great to build on the ensemble mean analysis to look at extreme events, following the kind of analysis in our [working group paper](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-230/). In other words, how does stratospheric nudging impact the probability of extremes in ACE2?
