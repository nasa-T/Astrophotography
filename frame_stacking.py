import rawpy
from PIL import Image, ExifTags
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear, demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007
import numpy as np
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder
import os
from astropy.io import fits
import pickle as p
from light_frame_processing import *
from photutils.aperture import CircularAperture
from scipy.ndimage import shift

def remove_from_list(l, e):
    if type(e) not in [list, tuple]:
        e = [e]
    new_l = []
    for i in l:
        if i not in e:
            new_l.append(i)
    return new_l

def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX, [0,0]], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def get_shifty(shift_dict, star_tabs, matches, ref=0):
    assert ref in shift_dict
    for j in matches[ref]:
        # print(f'check {j}')
        if j not in shift_dict:
            (a1, b1), (a2, b2) = matches[ref][j]
            a1 = star_tabs[ref][a1-1]
            b1 = star_tabs[ref][b1-1]
            a2 = star_tabs[j][a2-1]
            b2 = star_tabs[j][b2-1]
            if ((a1['flux']/b1['flux']) > 1 and (a2['flux']/b2['flux']) < 1) or ((a1['flux']/b1['flux']) < 1 and (a2['flux']/b2['flux']) > 1): # if a1 matches better with b2 then switch
                a2, b2 = b2, a2
            x_shift, y_shift = (a1['xcentroid'] - a2['xcentroid'], a1['ycentroid'] - a2['ycentroid'])
            star_dist = np.sqrt((a1['xcentroid'] - b1['xcentroid'])**2 + (a1['ycentroid'] - b1['ycentroid'])**2)
            im_dist = np.sqrt((b1['xcentroid'] - b2['xcentroid']+x_shift)**2 + (b1['ycentroid'] - b2['ycentroid']+y_shift)**2)
            rot = np.arcsin(im_dist/star_dist)
            if ref in shift_dict:
                x_shift, y_shift = shift_dict[ref][0]+x_shift, shift_dict[ref][1]+y_shift
            shift_dict[j] = (x_shift, y_shift, rot)
            # print(j)
    whats_left = remove_from_list(list(range(len(star_tabs))), list(shift_dict.keys()))
    if whats_left != []:
        for i in shift_dict:
            for j in whats_left:
                if j in matches[i]:  
                    return get_shifty(shift_dict, star_tabs, matches, i)
    return shift_dict

if __name__ == "__main__":
    dir_path = 'M81+82_052424/'
    light_files = [os.path.join(dir_path,f'DSC_{n:04}.NEF') for n in range(78,106)]
    light_exp = 150 # exposure time in seconds
    flat_exp = 12
    dark_exp = 150
    print('Reading Files...')
    light_frames = np.array([rawpy.imread(file).raw_image for file in light_files])
    master_bias = fits.open('master_bias.fits')[0].data
    master_dark = fits.open('master_dark.fits')[0].data
    master_flat = fits.open('master_flat.fits')[0].data

    with open('light_frames_sections.p', 'rb') as f:
        rgb_lights = p.load(f)
    
    print('Calibrating Frames...')
    processed_lights = []
    for i, frame in enumerate(light_frames):
        frame = frame - master_bias - master_dark*light_exp
        frame /= master_flat
        processed_lights.append(frame)

    

    print('Finding Stars...')
    star_tables = []
    for light in rgb_lights:
        f = light[:,:,1]
        finder = DAOStarFinder(12000,10,brightest=3)
        star_tables.append(finder.find_stars(f))

    dist_map_list = []
    for tab in star_tables:
        dist_map = {}
        for i, star in enumerate(tab):
            for star2 in tab[i+1:]:
                dist_map[(star['id'], star2['id'])] = np.sqrt((star['xcentroid'] - star2['xcentroid'])**2 + (star['ycentroid'] - star2['ycentroid'])**2)
        dist_map_list.append(dist_map)

    ref = [] # index of reference frame
    max_count = 0
    matched_dicts = {}
    print('Finding Matches...')
    for i, d in enumerate(dist_map_list):
        pair_dict = {}
        match_count = 0
        # d = dist_map_list[ref]
        for j, d2 in enumerate(dist_map_list):
            for pair in d:
                dist = np.round(d[pair],0)
                for pair2 in d2:
                    if np.round(d2[pair2],0) == dist:
                        # print(f'found pair of pairs: {pair} and {pair2}')
                        pair_dict[j] = (pair, pair2)
                        match_count+=1
                        break
                if np.round(d2[pair2],0) == dist:
                    break
        matched_dicts[i] = pair_dict
        # matched_list.append(pair_dict)
        if match_count > max_count:
            max_count = match_count
            ref = [i]
        elif match_count == max_count:
            ref += [i]
    print(f'best ref(s) is/are {ref} with {max_count} matches')

    shifts = {ref[0]:(0,0,0)}
    shifts = get_shifty(shifts,star_tables, matched_dicts, 10)
    idxs = list(shifts.keys())
    shifted_lights = {}
    with Bar("Aligning...", max=len(idxs)) as bar:
        for i in idxs:
            shifted_lights[i] = shift(processed_lights[i], (shifts[i][1], shifts[i][0]))
            bar.next()
    # [shift(processed_lights[i], (shifts[i][1], shifts[i][0])) for i in idxs]
    proceed = False
    remd_ims = [25,24,23,6]
    while not proceed:
        user = input(f"Choose which number image to view ({0}-{len(light_files)-1})\nOr type 'c' to continue: ")
        if user == 'c':
            proceed = True
        elif int(user) in shifted_lights.keys():
            plt.imshow(shifted_lights[int(user)])
            plt.show()
            user2 = input("Remove Image? (y/n)")
            if user2 == 'y':
                remd_ims.append(int(user))
        else:
            print('Alignment for the given image has failed.')
    idxs = remove_from_list(idxs, remd_ims)
    to_stack = np.array([shifted_lights[i] for i in idxs])
    stacked = np.median(to_stack, axis=0)
    print('Stacked')
    hdu = fits.PrimaryHDU(stacked)
    hdu.writeto('stacked_image.fits')
    plt.imshow(stacked)
    