from demcoreg import dem_align as da

src_dems = [#r"Z:\ATD\Drone Data Processing\Exports\Differencing_Testing\Clipped LIDAR Test\Mod_CoReg_All____MM090122_DEM_masked_1.tif",
            r"Z:\ATD\Drone Data Processing\Exports\Bennett\LIDAR\092021_LIDAR.tin.tif"
            ]    
    
for src in src_dems:    
    kwargs = {   
    'ref_dem_fn': r"Z:\ATD\Drone Data Processing\Exports\Bennett\LIDAR\November2013-April2014_LIDAR.tin.tif",
    'src_dem_fn': src,
    'mode': 'nuth',
    'mask_list': [],
    'max_offset': 100,
    'max_dz': 100,
    'slope_lim': (0.1, 40),
    'tiltcorr': False,
    'polyorder': 1,
    'res': 'max',
    'max_iter': 30,
    'tol': 0.02,
    'outdir': r"Z:\ATD\Drone Data Processing\Exports\Differencing_Testing\Clipped LIDAR Test"
    }
    da.dem_align(**kwargs)