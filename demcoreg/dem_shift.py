import os
from osgeo import gdal
from pygeotools.lib import iolib, warplib, geolib
from demcoreg import coreglib 
from demcoreg import dem_align as da
import numpy as np
import matplotlib.pyplot as plt
import sys
from pygeotools.lib import iolib, malib, geolib, warplib, filtlib


def get_shift(ref_dem_fn, src_dem_fn, outdir, mode ='nuth', res ='max', **kwargs):
    #parser = getparser()
    #args = parser.parse_args()

    ref_dem_fn = ref_dem_fn
    src_dem_fn = src_dem_fn
    outdir = outdir
    mode = mode
    res = res
    mask_list = kwargs.get('mask_list', [])
    max_offset = kwargs.get('max_offset', 100)
    max_dz = kwargs.get('max_dz', 100)
    slope_lim = kwargs.get('slope_lim', (0, 50))
    tiltcorr = kwargs.get('tiltcorr', False)
    polyorder = kwargs.get('polyorder', 1)
    
    max_iter = kwargs.get('max_iter', 30)
    tol = kwargs.get('tol', 0.005)

    min_dx = tol
    min_dy = tol
    min_dz = tol
    
    if outdir is None:
        outdir = os.path.splitext(src_dem_fn)[0] + '_dem_align'

    if tiltcorr:
        outdir += '_tiltcorr'
        tiltcorr_done = False


    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outprefix = '%s_%s' % (os.path.splitext(os.path.split(src_dem_fn)[-1])[0], \
            os.path.splitext(os.path.split(ref_dem_fn)[-1])[0])
    outprefix = os.path.join(outdir, outprefix)

    print("\nReference: %s" % ref_dem_fn)
    print("Source: %s" % src_dem_fn)
    print("Mode: %s" % mode)
    print("Output: %s\n" % outprefix)
    print("Tolerance: %0.3f" % tol)
    print("Max iterations: %i" % max_iter)
    print("Slope limits: %0.2f - %0.2f" % slope_lim)
    src_dem_ds = gdal.Open(src_dem_fn)
    ref_dem_ds = gdal.Open(ref_dem_fn)

    #Define local cartesian coordinate system  
    #Should compute equidistant projection based on clon,clat and extent
    #local_srs = geolib.localtmerc_ds(src_dem_ds)

    #Use original source dataset coordinate system
    #Potentially issues with distortion and xyz/tiltcorr offsets for DEMs with large extent
    src_dem_srs = geolib.get_ds_srs(src_dem_ds)
    ref_dem_srs = geolib.get_ds_srs(ref_dem_ds)
    local_srs = src_dem_srs
    #local_srs = ref_dem_srs 

    #Check that inputs are projected CRS with units of meters
    #In principle, this should only be required for the src_dem (local_srs above), as the ref_dem will be reprojected to match
    #In practice, sometimes the intersection can fail for small extents, so best to reproject both
    if not src_dem_srs.IsProjected() or not ref_dem_srs.IsProjected():
        #New function added to geolib 8/22/23
        #epsg = geolib.getUTMepsg(geolib.ds_geom(src_dem_ds))
        #Copied here to avoid version issues
        utm = geolib.getUTMzone(geolib.ds_geom(src_dem_ds))
        prefix = '326'
        if utm[-1] == 'S':
            prefix = '327'
        epsg = prefix+utm[0:2]

        print(f"{ref_dem_fn} CRS: '{ref_dem_srs.ExportToProj4()}'")
        print(f"{src_dem_fn} CRS: '{src_dem_srs.ExportToProj4()}'\n")

        print(f"Input DEMs must have projected CRS with linear units of meters (not WGS84 geographic, units of degrees)") 
        print(f"For relatively limited DEM extents (<100-300 km), you can consider reprojecting using the appropriate UTM projection:")
        print(f"gdalwarp -r cubic -t_srs EPSG:{epsg} {src_dem_fn} {os.path.splitext(src_dem_fn)[0]+'_proj.tif'}")
        print("For larger DEM extents, consider a custom equidistant projection: https://projectionwizard.org/")
        print(f"Then rerun the dem_align.py command with the projected DEM(s)\n")

        sys.exit()

    #Resample to common grid
    ref_dem_res = float(geolib.get_res(ref_dem_ds, t_srs=local_srs, square=True)[0])
    #Create a copy to be updated in place
    src_dem_ds_align = iolib.mem_drv.CreateCopy('', src_dem_ds, 0)
    src_dem_res = float(geolib.get_res(src_dem_ds, t_srs=local_srs, square=True)[0])
    src_dem_ds = None
    #Resample to user-specified resolution
    ref_dem_ds, src_dem_ds_align = warplib.memwarp_multi([ref_dem_ds, src_dem_ds_align], \
            extent='intersection', res=res, t_srs=local_srs, r='cubic')

    res = float(geolib.get_res(src_dem_ds_align, square=True)[0])
    print("\nReference DEM res: %0.2f" % ref_dem_res)
    print("Source DEM res: %0.2f" % src_dem_res)
    print("Resolution for coreg: %s (%0.2f m)\n" % (res, res))

    #Iteration number
    n = 1
    #Cumulative offsets
    dx_total = 0
    dy_total = 0
    dz_total = 0

    #Now iteratively update geotransform and vertical shift
    while True:
        print("*** Iteration %i ***" % n)
        dx, dy, dz, static_mask, fig = da.compute_offset(ref_dem_ds, src_dem_ds_align, src_dem_fn, mode, max_offset, \
                mask_list=mask_list, max_dz=max_dz, slope_lim=slope_lim, plot=True)
        xyz_shift_str_iter = "dx=%+0.2fm, dy=%+0.2fm, dz=%+0.2fm" % (dx, dy, dz)
        print("Incremental offset: %s" % xyz_shift_str_iter)

        dx_total += dx
        dy_total += dy
        dz_total += dz

        xyz_shift_str_cum = "dx=%+0.2fm, dy=%+0.2fm, dz=%+0.2fm" % (dx_total, dy_total, dz_total)
        print("Cumulative offset: %s" % xyz_shift_str_cum)
        #String to append to output filenames
        xyz_shift_str_cum_fn = '_%s_x%+0.2f_y%+0.2f_z%+0.2f' % (mode, dx_total, dy_total, dz_total)

        #Should make an animation of this converging
        if n == 1: 
            #static_mask_orig = static_mask
            if fig is not None:
                dst_fn = outprefix + '_%s_iter%02i_plot.png' % (mode, n)
                print("Writing offset plot: %s" % dst_fn)
                fig.gca().set_title("Incremental: %s\nCumulative: %s" % (xyz_shift_str_iter, xyz_shift_str_cum))
                fig.savefig(dst_fn, dpi=300)

        #Apply the horizontal shift to the original dataset
        src_dem_ds_align = coreglib.apply_xy_shift(src_dem_ds_align, dx, dy, createcopy=False)
        #Should 
        src_dem_ds_align = coreglib.apply_z_shift(src_dem_ds_align, dz, createcopy=False)

        n += 1
        print("\n")
        #If magnitude of shift in all directions is less than tol
        #if n > max_iter or (abs(dx) <= min_dx and abs(dy) <= min_dy and abs(dz) <= min_dz):
        #If magnitude of shift is less than tol
        dm = np.sqrt(dx**2 + dy**2 + dz**2)
        dm_total = np.sqrt(dx_total**2 + dy_total**2 + dz_total**2)

        if dm_total > max_offset:
            sys.exit("Total offset exceeded specified max_offset (%0.2f m). Consider increasing -max_offset argument" % max_offset)

        #Stop iteration
        if n > max_iter or dm < tol:

            if fig is not None:
                dst_fn = outprefix + '_%s_iter%02i_plot.png' % (mode, n)
                print("Writing offset plot: %s" % dst_fn)
                fig.gca().set_title("Incremental:%s\nCumulative:%s" % (xyz_shift_str_iter, xyz_shift_str_cum))
                fig.savefig(dst_fn, dpi=300)

            #Compute final elevation difference
            if True:
                ref_dem_clip_ds_align, src_dem_clip_ds_align = warplib.memwarp_multi([ref_dem_ds, src_dem_ds_align], \
                        res=res, extent='intersection', t_srs=local_srs, r='cubic')
                ref_dem_align = iolib.ds_getma(ref_dem_clip_ds_align, 1)
                src_dem_align = iolib.ds_getma(src_dem_clip_ds_align, 1)
                ref_dem_clip_ds_align = None

                diff_align = src_dem_align - ref_dem_align
                src_dem_align = None
                ref_dem_align = None

                #Get updated, final mask
                static_mask_final = da.get_mask(src_dem_clip_ds_align, mask_list, src_dem_fn)
                static_mask_final = np.logical_or(np.ma.getmaskarray(diff_align), static_mask_final)
                
                #Final stats, before outlier removal
                diff_align_compressed = diff_align[~static_mask_final]
                diff_align_stats = malib.get_stats_dict(diff_align_compressed, full=True)

                #Prepare filtered version for tiltcorr fit
                diff_align_filt = np.ma.array(diff_align, mask=static_mask_final)
                diff_align_filt = da.outlier_filter(diff_align_filt, f=3, max_dz=max_dz)
                #diff_align_filt = outlier_filter(diff_align_filt, perc=(12.5, 87.5), max_dz=max_dz)
                slope = da.get_filtered_slope(src_dem_clip_ds_align)
                diff_align_filt = np.ma.array(diff_align_filt, mask=np.ma.getmaskarray(slope))
                diff_align_filt_stats = malib.get_stats_dict(diff_align_filt, full=True)

            #Fit 2D polynomial to residuals and remove
            #To do: add support for along-track and cross-track artifacts
            if tiltcorr and not tiltcorr_done:
                print("\n************")
                print("Calculating 'tiltcorr' 2D polynomial fit to residuals with order %i" % polyorder)
                print("************\n")
                gt = src_dem_clip_ds_align.GetGeoTransform()

                #Need to apply the mask here, so we're only fitting over static surfaces
                #Note that the origmask=False will compute vals for all x and y indices, which is what we want
                vals, resid, coeff = geolib.ma_fitpoly(diff_align_filt, order=polyorder, gt=gt, perc=(0,100), origmask=False)
                #vals, resid, coeff = geolib.ma_fitplane(diff_align_filt, gt, perc=(12.5, 87.5), origmask=False)

                #Should write out coeff or grid with correction 

                vals_stats = malib.get_stats_dict(vals)

                #Want to have max_tilt check here
                #max_tilt = 4.0 #m
                #Should do percentage
                #vals.ptp() > max_tilt

                #Note: dimensions of ds and vals will be different as vals are computed for clipped intersection
                #Need to recompute planar offset for full src_dem_ds_align extent and apply
                xgrid, ygrid = geolib.get_xy_grids(src_dem_ds_align)
                valgrid = geolib.polyval2d(xgrid, ygrid, coeff) 
                #For results of ma_fitplane
                #valgrid = coeff[0]*xgrid + coeff[1]*ygrid + coeff[2]
                src_dem_ds_align = coreglib.apply_z_shift(src_dem_ds_align, -valgrid, createcopy=False)

                print("Applying tilt correction to difference map")
                diff_align -= vals

                #Should iterate until tilts are below some threshold
                #For now, only do one tiltcorr
                tiltcorr_done=True
                #Now use original tolerance, and number of iterations 
                tol = tol
                max_iter = n + max_iter
            else:
                break
    
    align_fn = outprefix + '%s_align.tif' % xyz_shift_str_cum_fn
    print("Writing out shifted src_dem with median vertical offset removed: %s" % align_fn)
    #Open original uncorrected dataset at native resolution
    src_dem_ds = gdal.Open(src_dem_fn)
    src_dem_ds_align = iolib.mem_drv.CreateCopy('', src_dem_ds, 0)
    #Apply final horizontal and vertial shift to the original dataset
    #Note: potentially issues if we used a different projection during coregistration!
    src_dem_ds_align = coreglib.apply_xy_shift(src_dem_ds_align, dx_total, dy_total, createcopy=False)
    src_dem_ds_align = coreglib.apply_z_shift(src_dem_ds_align, dz_total, createcopy=False)
    if tiltcorr:
        xgrid, ygrid = geolib.get_xy_grids(src_dem_ds_align)
        valgrid = geolib.polyval2d(xgrid, ygrid, coeff) 
        #For results of ma_fitplane
        #valgrid = coeff[0]*xgrid + coeff[1]*ygrid + coeff[2]
        src_dem_ds_align = coreglib.apply_z_shift(src_dem_ds_align, -valgrid, createcopy=False)
    #Might be cleaner way to write out MEM ds directly to disk
    src_dem_full_align = iolib.ds_getma(src_dem_ds_align)
    iolib.writeGTiff(src_dem_full_align, align_fn, src_dem_ds_align)
    return align_fn, [dx_total, dy_total, dz_total] 

def dem_shift(src_dem_fn, outdir, shift):
    """
    Shifts a Digital Elevation Model (DEM) by a specified amount in the x, y, and z directions using pygeotools library functions.
    Parameters:
    - src_dem_fn (str): The file path of the source DEM to be shifted.
    - outdir (str): The directory where the shifted DEM will be saved as a GeoTIFF file.
    - shift (tuple): The amount to shift the DEM in the x, y, and z directions.
    Returns:
    - src_out_fn (str): The file path for the shifted DEM GeoTIFF.
    """
    dx = shift[0]
    dy = shift[1]
    dz = shift[2]
    src_dem_ds = gdal.Open(src_dem_fn)
    print(f"Shifting DEM by: {dx}, {dy}, {dz}")
    src_dem_ds_align = iolib.mem_drv.CreateCopy('', src_dem_ds, 0)
    if dx is not None and dy is not None and dz is not None:
        #Apply the horizontal shift to the original dataset
        print("Applying xy shift: %0.2f, %0.2f" % (dx, dy))
        src_dem_ds_align = coreglib.apply_xy_shift(src_dem_ds_align, dx, dy, createcopy=False)
        print("Applying z shift: %0.2f" % dz)
        src_dem_ds_align = coreglib.apply_z_shift(src_dem_ds_align, dz, createcopy=False)
    
    print("Converting source DEM to array...")
    src_dem_full_align = iolib.ds_getma(src_dem_ds_align)
    src_out_fn = os.path.join(outdir, os.path.splitext(os.path.split(src_dem_fn)[-1])[0] + '_shifted.tif')
    print("Writing source DEM to: %s" % src_out_fn)
    iolib.writeGTiff(src_dem_full_align, src_out_fn, src_dem_ds_align)
    return src_out_fn
    