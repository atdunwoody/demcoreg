from demcoreg import dem_align as da

src_dems = [#r"Z:\ATD\Drone Data Processing\Exports\Differencing_Testing\Clipped LIDAR Test\Mod_CoReg_All____MM090122_DEM_masked_1.tif",
            r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LPM\07092023 Full Run\Masked_Veg_DEM_Logs.tif"

            ]    
    
for src in src_dems:    
    kwargs = {   
    'ref_dem_fn': r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LIDAR\Reprojected to UTM Zone 13N\ET_low_LIDAR_2020_1m_DEM_reproj.tif",
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