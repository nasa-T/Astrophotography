import rawpy
from PIL import Image, ExifTags
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear, demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007
import numpy as np
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder
import os
from astropy.io import fits
import pickle as p
from progress.bar import Bar

def calibrate_light_frames(frames, mbias, mdark, mflat, light_exp):
    processed_lights = []
    with Bar("Calibrating...", max=len(frames)) as bar:
        for frame in frames:
            frame = frame - mbias - mdark*light_exp
            frame /= mflat
            processed_lights.append(frame)
            bar.next()
    return processed_lights

def debayer_frames(frames, keep_portion=True):
    rgb_lights = []
    # print("Demosaicing...")
    with Bar("Demosaicing...", max=len(frames)) as bar:
        count = 0
        for frame in frames:
            if keep_portion:
                frame = frame[1000:3000,1500:2500]
            rgb_lights.append(demosaicing_CFA_Bayer_Menon2007(frame, 'RGGB'))
            # print(str(100*count/len(frames))+'%')
            count+=1
            bar.next()
    return rgb_lights

if __name__ == "__main__":
    dir_path = 'M81+82_052424/'
    light_files = [os.path.join(dir_path,f'DSC_{n:04}.NEF') for n in range(78,106)]
    light_exp = 150 # exposure time in seconds
    flat_exp = 12
    dark_exp = 150

    light_frames = np.array([rawpy.imread(file).raw_image for file in light_files])[:7]
    master_bias = fits.open('master_bias.fits')[0].data
    master_dark = fits.open('master_dark.fits')[0].data
    master_flat = fits.open('master_flat.fits')[0].data
    clights = calibrate_light_frames(light_frames, master_bias, master_dark, master_flat, light_exp)
    print('Light Frames Calibrated')
    rgb = debayer_frames(clights, False)
    print('Pickling...')
    with open('light_frames.p', 'wb') as f:
        p.dump(rgb, f)
    # star_tables = 
    
