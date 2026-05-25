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
import gc

def calibrate_light_frames(frames, mbias=0, mdark=0, mflat=1):
    processed_lights = []
    with Bar("Calibrating...", max=len(frames)) as bar:
        for frame in frames:
            frame = frame - mbias - mdark
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
            #rgb_lights.append(demosaicing_CFA_Bayer_Menon2007(frame, 'RGGB'))
            rgb_lights.append(demosaicing_CFA_Bayer_bilinear(frame, 'RGGB'))
            # print(str(100*count/len(frames))+'%')
            count+=1
            bar.next()
    return rgb_lights

if __name__ == "__main__":
    dir_path = '/home/Tasan/astrophotos_051726/cygnus'
    light_files = [os.path.join(dir_path, im) for im in os.listdir(dir_path)]
    light_exp = 90 # exposure time in seconds
    flat_exp = 12
    dark_exp = 30
    print(light_files)
    light_frames = np.array([rawpy.imread(file).raw_image for file in light_files], dtype=np.float16)
    #master_bias = fits.open('master_bias.fits')[0].data
    master_dark = fits.open('../master_dark_30s.fits')[0].data.astype(np.float16)
    #master_flat = fits.open('master_flat.fits')[0].data
    clights = np.array(calibrate_light_frames(light_frames, mdark=master_dark),dtype=np.float16)

    # For memory management, divide into 20-image sections
    for i in range(len(clights)//20):
        rgb = debayer_frames(clights[i*20:(i+1)*20], False)
    
        print('Pickling...')
        with open(f'light_frames_{i}.p', 'wb') as f:
            p.dump(rgb, f)

    if len(clights)%20 != 0:
        #print('Light Frames Calibrated')
        rgb = debayer_frames(clights[len(clights)//20*20:], False)
        
        print('Pickling...')
        with open(f'light_frames_{len(clights)//20}.p', 'wb') as f:
            p.dump(rgb, f)
    # star_tables = 
    
