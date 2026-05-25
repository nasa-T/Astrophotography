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
import glob

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
    #print(shift_dict, ref)
    assert ref in shift_dict
    for j in matches[ref]:
        # print(f'check {j}')
        if j not in shift_dict:
            (a1, b1), (a2, b2) = matches[ref][j]
            a1 = star_tabs[ref][a1-1]
            b1 = star_tabs[ref][b1-1]
            a2 = star_tabs[j][a2-1]
            b2 = star_tabs[j][b2-1]
            ref_dir = (np.atan2((a1['y_centroid'] - b1['y_centroid']),(a1['x_centroid'] - b1['x_centroid']))+2*np.pi)%(2*np.pi)
            j_dir = (np.atan2((a2['y_centroid'] - b2['y_centroid']),(a2['x_centroid'] - b2['x_centroid']))+2*np.pi)%(2*np.pi)
            
            if np.abs(ref_dir - j_dir) > np.pi/2:
            #if ((a1['flux']/b1['flux']) > 1 and (a2['flux']/b2['flux']) < 1) or ((a1['flux']/b1['flux']) < 1 and (a2['flux']/b2['flux']) > 1): # if a1 matches better with b2 then switch
                a2, b2 = b2, a2
                
            x_shift, y_shift = (a1['x_centroid'] - a2['x_centroid'], a1['y_centroid'] - a2['y_centroid'])
            star_dist = np.sqrt((a1['x_centroid'] - b1['x_centroid'])**2 + (a1['y_centroid'] - b1['y_centroid'])**2)
            im_dist = np.sqrt((b1['x_centroid'] - b2['x_centroid']+x_shift)**2 + (b1['y_centroid'] - b2['y_centroid']+y_shift)**2)
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
    with open('light_frames.p', 'rb') as f:
        rgb_lights = np.array(p.load(f), dtype=np.float16)
    
    print('Finding Stars...')
    star_tables = []
    for light in rgb_lights:
        f = light[:,:,1]
        finder = DAOStarFinder(12000,10,n_brightest=5)
        star_tables.append(finder.find_stars(f.astype(np.float32)))

    dist_map_list = []
    for tab in star_tables:
        dist_map = {}
        for i, star in enumerate(tab):
            for star2 in tab[i+1:]:
                dist_map[(star['id'], star2['id'])] = np.sqrt((star['x_centroid'] - star2['x_centroid'])**2 + (star['y_centroid'] - star2['y_centroid'])**2)
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
    shifts = get_shifty(shifts,star_tables, matched_dicts, ref[0])
    idxs = list(shifts.keys())
    print(idxs)
    shifted_lights = {}
    with Bar("Aligning...", max=len(idxs)) as bar:
        for i in idxs:
            shifted_lights[i] = shift(rgb_lights[i].astype(np.float32), (shifts[i][1], shifts[i][0], 0)).astype(np.float16)
            bar.next()
            
    proceed = False
    remd_ims = []
    while not proceed:
        user = input(f"Choose which number image to view ({0}-{len(rgb_lights)-1})\nOr type 'c' to continue: ")
        if user == 'c':
            proceed = True
        elif int(user) in shifted_lights.keys():
            plt.imshow(shifted_lights[int(user)][:,:,1])
            plt.show()
            user2 = input("Remove Image? (y/n)")
            if user2 == 'y':
                remd_ims.append(int(user))
        else:
            print('Alignment for the given image has failed.')

    print("Removing images from stack.")
    idxs = remove_from_list(idxs, remd_ims)
    print("Stacking")
    to_stack = np.array([shifted_lights[i] for i in idxs])
    print(to_stack.max())
    stacked = np.median(to_stack, axis=0, overwrite_input=True)
    print('Stacked')
    hdu = fits.PrimaryHDU(stacked.astype(np.float32))
    hdu.writeto('stacked_image.fits', overwrite=True)
    plt.imshow((stacked-stacked.min())/(stacked.max()-stacked.min()))
    plt.show()
    
