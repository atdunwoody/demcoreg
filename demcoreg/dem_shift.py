import os
from osgeo import gdal
from pygeotools.lib import iolib, warplib, geolib
import coreglib 

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
    