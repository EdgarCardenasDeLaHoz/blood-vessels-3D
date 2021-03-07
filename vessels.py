import os 
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import h5py

from scipy import ndimage as ndi
from scipy.interpolate import interpn

from skimage import io,morphology
from skimage.filters import gaussian

import skan

from matplotlib import collections  as mc

###########################
#  Entry Points
##########################

def process_folder(path):
    
    fns = glob.glob(path + "*.tif")
    data_out = []
    for fn in fns:  
        print(fn)  
        df_vessel = analyze_BV(fn)
        data_out.append( df_vessel)
        plt.close("all")

    all_df = pd.concat(data_out)
    xls_fn = path + "\\Output\\Summary_data.xls"
    all_df.to_excel(xls_fn)

    summary_figure(all_df,path)

def analyze_BV(fn):

    ## Input
    im = read_data_BV(fn)
    save_dataset(fn, im )

    ## Process
    im = downsample(im,scale=0.5)
    im1,d_seg,skel = segment_BV(im)    
    d_skel = d_seg*skel
    b_seg =  d_seg>0

    G_skel = skan.Skeleton(d_skel)

    draw_graph_2D(fn, G_skel)

    ## Output 
    images = (im, im1, d_seg, skel)
    render_videos(fn, images, G_skel)
   
    df_vessel = measure_vessel(fn, G_skel, b_seg)

    return df_vessel

###########################
#     Data IO 
############################

def read_data_BV(fn):

    ## output dimensions should be 1,Z,X,Y
    im = io.imread(fn)

    if im.ndim==4:
        im = im[...,0]

    if im.ndim==3:
        im = im[None,...]

    return im 

def save_h5(fn,im):   
    with h5py.File(fn, 'a') as fh:
        if "image_data" in fh.keys():   del fh["image_data"]
        fh.create_dataset("im_data", data=im , compression="gzip",compression_opts=1)

def save_dataset(fn_in,im):

    fileparts = fn_in.split("\\")
    folder = ("\\").join(fileparts[:-1]) + "\\Dataset\\"
    if not os.path.exists(folder): os.makedirs(folder)
    fn_out = folder + fileparts[-1]
    fn_out = fn_out.replace(".tif",".H5")

    if not os.path.exists(fn_out): 
        save_h5(fn_out,im)

##########################
#      Image Processing
###########################

def segment_BV(im):
    
    im1 = im.copy()
    #im_min = np.mean(im1,axis=(1,2,3))[:,None,None,None]
    ####

    im_min = ndi.uniform_filter( im1, size=(1,50,50))
    im1 = im1 - im_min
    im1[im1<0] = 0

    im_max = np.array([np.quantile(I[I>0],.9) for I in im1])
    im1 = im1 / im_max[:,None,None]
    im1[im1>1] = 1

    #####

    im1 = (im1*255).astype(np.uint8)
    im1 = ndi.uniform_filter( im1, size=(3,5,5))
    
    seg = im1>50

    seg = area_filter(seg, min_size=500)>0
    #seg = morphology.binary_closing(seg,np.ones((3,5,5)))
    seg = morphology.remove_small_holes(seg, 5000)
    skel  = morphology.skeletonize(seg)*1

    dist = np.array(ndi.distance_transform_edt(seg,[5,1,1]) )

    _, inds = ndi.distance_transform_edt(~skel, sampling=[5,1,1], return_indices=True)
    d_seg = dist[inds[0],inds[1],inds[2]]    

    d_seg[~seg] = 0

    return im1,d_seg,skel

def area_filter(ar, min_size=0, max_size=None):
    """
    """
    if ar.dtype == bool:
        ccs,l_max = ndi.label(ar)
    else:
        ccs = ar
        l_max = ar.max()

    component_sizes = np.bincount(ccs[ccs>0])
    idxs = np.arange(l_max+1).astype(np.uint16)
    if min_size>0:
        too_small = component_sizes < min_size
        idxs[too_small]=0

    if max_size is not None:
        too_large = component_sizes > max_size
        idxs[too_large]=0

    out = np.zeros_like(ccs, np.uint16)
    _, idxs2 = np.unique(idxs,return_inverse=True)
    out[ccs>0] = idxs2[ccs[ccs>0]]

    return out

def downsample(im,scale=0.5):

    im = im[0]
    sz = tuple((np.array(im.shape)[[1,2]]*scale).astype(np.int))
    im = np.array([cv2.resize(I,sz) for I in im])
    im = im/255

    return im 

##################################
##         Measurement
##################################

def measure_vessel(fn, G_skel, b_seg,):

    df_skel = skan.summarize(G_skel)

    df_skel["tortuosity"] = df_skel['euclidean-distance'] / df_skel['branch-distance']
    print_vessel_measurements(df_skel)

    df_vessel = pd.DataFrame()
    df_vessel["N_Segments"] = [len(df_skel)]
    df_vessel["N_Endpoints"] = [sum(df_skel["branch-type"]==1)]
    df_vessel["N_Branchpoints"] =  [sum(df_skel["branch-type"]==2)]

    df_vessel["Total_VesselLength"] =  df_skel["branch-distance"].sum()
    df_vessel["Mean_BranchLength"] = df_skel["branch-distance"].mean()
    df_vessel["Mean_BranchThickness"] = df_skel["mean-pixel-value"].mean()
    
    df_vessel["Max_BranchLength"] = df_skel["branch-distance"].max()
    df_vessel["Max_BranchThickness"] = df_skel["mean-pixel-value"].max() 
    df_vessel["Mean_Tortuosity"] = df_skel["tortuosity"].mean()

    df_vessel["Vessel_Solidity"] = [b_seg.mean()*100]
    df_vessel["filename"] = fn.split("\\")[-1]

    fn_out = out_name(fn)+ "_figures.png"
    plt.savefig(fn_out)
    plt.close("all")

    return df_vessel

#################################
#    Analysis Writing
#################################

def print_vessel_measurements(df_skel):

    fig, axs = plt.subplots(2,2, figsize= (14,8))

    columns = ["branch-distance", "mean-pixel-value"]
    col_names = ["Branch_Length", "Branch_Thickness"]
    for n,col in enumerate(columns):
        ax = axs.ravel()[n]
        ax.hist(df_skel[col],100)
        ax.set_title(col_names[n])
        ax.set_ylabel("n")
        ax.grid(True)

    x = df_skel["branch-distance"][::10]
    y = df_skel[ "mean-pixel-value"][::10]
    axs[1,0].set_xlabel("Branch_Length (pixels)")
    axs[1,0].set_ylabel("Thickness (pixels)")
    axs[1,0].plot(x,y,".")
    axs[1,0].grid(True)

    x = df_skel["branch-distance"][::10]
    y = df_skel[ "tortuosity"][::10]
    axs[1,1].set_xlabel("Branch_Length (pixels)")
    axs[1,1].set_ylabel("Tortuosity")
    axs[1,1].plot(x,y,".")
    axs[1,1].grid(True)

def summary_figure(all_df, path):

    columns = all_df.columns[:-1]
    fig,axs = plt.subplots(2,5,figsize=(20,12))
    labels = ["ctrl","ctrl","ctrl","MCD","MCD","MCD"]
    for n,col in enumerate(columns):
        ax = axs.ravel()[n]
        x = np.array(range(len(all_df)))
        y = np.array(all_df[col])
        ax.plot(x,y,"o")
        ax.set_title(col)
        ax.grid(True)
        ax.set_xticks( x )
        ax.set_xticklabels( labels)
    plt.savefig(path + "\\Output\\Summary_data.jpg") 

def out_name(fn_in):
    
    fileparts = fn_in.split("\\")

    folder = ("\\").join(fileparts[:-1]) + "\\Output\\"
    if not os.path.exists(folder): os.makedirs(folder)
    fn_out = folder + fileparts[-1]
    
    fn_out = fn_out.replace(".tif","")

    return fn_out
    
def write_figure(fn, images):
    
    fn_out = fn + "_output.jpg"
    im1,im2,im3 = images
    
    draw = np.hstack((im1.mean(axis=0),im2.mean(axis=0),im3.mean(axis=0)))
    
    draw = draw.astype(np.uint8)
    plt.imsave(fn_out,draw)

    plt.figure()
    plt.imshow(draw,cmap="jet")

#################################
#       3D Spin Rendering
#################################

def render_videos( fn, images, G_skel ):

    #images = im, im1, d_seg, skel
    color_ims = colorize_images(images)

    frames = [] 
    for i in [1,2,4]:
        c_im3 = color_ims[i].transpose(3,1,2,0)
        frames.append( collect_spin_frames(c_im3) )
   
    fn_out = out_name(fn)+ "_spin3D.mp4"

    frames = np.concatenate(frames)
    render_frames(fn_out , frames, FPS=5 )

    fn_out = out_name(fn)+ "_zbyz.mp4"
    frames = zxz_frames( color_ims )
    render_frames(fn_out , frames, FPS=5 )

def collect_spin_frames(Im3, n_t=6):

    ## Im3 dims : C,Z,Y,X
    if Im3.ndim==3:   Im3 = Im3[None,...]

    Im3 = Im3 / Im3.max()
    Im3 = (Im3*255).astype(np.uint8)

    angles = np.linspace(0,2*3.14,n_t)

    frs = []
    for a in angles:
        print(".",end="")
        Irot = proj_2D(Im3, a)
        frs.append(Irot)
    print("")

    frs = np.array(frs).transpose(0,2,3,1)
    frs = frs / frs.max()
    frs = (frs * 255).astype(np.uint8)
    return frs

def proj_2D( Im3, angle, method="mean"):

    out_sz = 200

    xi,yi,zi, qxR,qyR,qzR = rotate_image_space(Im3[0], angle, out_sz)

    I3_out = [] 

    for I in Im3:
        
        if np.sum(I)==0: 
            mat = np.zeros((out_sz,out_sz))
            I3_out.append(mat)
            continue 

        I_tf = np.zeros((out_sz, out_sz, 20))

        bI = I>0
        #I_tf[qxR,qyR,qzR] = I[xi,yi,zi]
        I_tf[qxR[bI],qyR[bI],qzR[bI]] = I[xi[bI],yi[bI],zi[bI]]

        #alpha = np.linspace(1,0.5,I_tf.shape[2])[None,None,:]
        #alpha = alpha / alpha.sum(axis=2) 
        #I_tf = I_tf*1.
        #I_tf = I_tf * alpha 

        if method=="mean":
            mat = np.mean(I_tf,axis=2)
            mat[mat>10] = 10
            mat = mat*25

        elif method == "max":
            mat = np.max(I_tf,axis=2)

        I3_out.append(mat)

    I3_out = np.array(I3_out)

    return I3_out

def rotate_image_space(I, angle, out_sz=100):

    Nx, Ny, Nz = I.shape

    xi = np.arange( Nx )
    yi = np.arange( Ny ) 
    zi = np.arange( Nz )

    zi = zi.astype(np.int)

    xi, yi, zi = np.meshgrid(xi,yi,zi)
    Nsz = np.max([Nx,Ny,Nz])

    qx = xi - Nx/2
    qy = yi - Ny/2
    qz = zi - Nz/2

    qx = qx/(Nsz-1) * 1.9
    qy = qy/(Nsz-1) * 1.9
    qz = qz/(Nsz-1) * 1.9 * 2 
   
    qxR,qyR,qzR = rotate_pts(qx,qy,qz,angle,out_sz)

    qxR = qxR.astype(np.int)
    qyR = qyR.astype(np.int)
    qzR = qzR.astype(np.int) 

    return xi,yi,zi, qxR,qyR,qzR

def rotate_pts(qx,qy,qz,angle,out_sz=100):

    x_sz = out_sz
    y_sz = out_sz 
    z_sz = 20 

    # query points
    qxR = qx 
    qyR = qy * np.cos(angle) - qz * np.sin(angle) 
    qzR = qy * np.sin(angle) + qz * np.cos(angle)

    qxR = qxR * 0.5 * x_sz / 1.2  * .98
    qyR = qyR * 0.5 * y_sz / 1.42 * .98
    qzR = qzR * 0.5 * z_sz / 1.42 * .98

    qxR = qxR + x_sz * .5
    qyR = qyR + y_sz * .5
    qzR = qzR + z_sz * .5

    return qxR,qyR,qzR

def proj_2D_interp(datacube, angle, xy_res=200, z_scale = 0.5, method="max"):
    
    nZ = 50 
    N = xy_res  # camera grid resolution
    #datacube = datacube
    pts = x_points(datacube,N, z_scale=z_scale)
    qi = q_points(N, angle, nZ=nZ)
    # Interpolate onto Camera Grid
    camera_grid = interpn(pts, datacube, qi, method='linear', bounds_error=False, fill_value=0)
    camera_grid = camera_grid.reshape((N,N,nZ))

    if method=="mean":
        mat = np.mean(camera_grid,axis=2)
    elif method == "max":
        mat = np.max(camera_grid,axis=2)

    return mat

def x_points(datacube, N, z_scale=0.2):
    
    Nx, Ny, Nz = datacube.shape
    x = np.linspace( -1, 1, Nx)
    y = np.linspace( -1, 1, Ny)
    z = np.linspace( -1, 1, Nz)
    points = (x,y,z)
    return points

def q_points(N, angle, nZ=50):
    # Construct the Camera Grid / Query Points -- rotate camera view
    xi = np.linspace(-1.1, 1.1, N)
    yi = np.linspace(-1.1, 1.1, N)
    zi = np.linspace(-1.1, 1.1, nZ)
    
    qx, qy, qz = np.meshgrid(xi,yi,zi)  # query points
    qxR = qx * np.cos(angle) - qz * np.sin(angle) 
    qyR = qy 
    qzR = qx * np.sin(angle) + qz * np.cos(angle)
    #qxR = qxR * (qzR*0.5 + 1.2)
    #qyR = qyR * (qzR*0.5 + 1.2)
    qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
    return qi

###############################
#  Plt plotting of the graph 
###############################

def draw_graph_2D(fn, G_skel):
    
    pts = G_skel.coordinates
    p_type = G_skel.degrees

    df_skel = skan.summarize(G_skel)

    segments = np.array(df_skel[["image-coord-src-0","image-coord-src-1","image-coord-src-2",
                                 "image-coord-dst-0","image-coord-dst-1","image-coord-dst-2"]])
    l_type = np.array(df_skel[["branch-type"]])[:,0]

    lines = segments[:, [1,2,4,5]]
    lines = lines.reshape(-1,2,2)

    fig,ax = plt.subplots(1,1, figsize=(18,18))

    l = 0
    plt.plot(pts[p_type==l,1] , pts[p_type==l,2],"or")
    l = 1
    plt.plot(pts[p_type==l,1] , pts[p_type==l,2],"og")
    l = 3
    plt.plot(pts[p_type==l,1] , pts[p_type==l,2],"ow")

    lc = mc.LineCollection(lines[l_type==1], colors=(1,1,0), linewidths=3)
    ax.add_collection(lc)
    lc = mc.LineCollection(lines[l_type==2], colors=(0,0,1), linewidths=3)
    ax.add_collection(lc)

    ax.set_facecolor("k")
    ax.grid(True)
    
    fn_out = out_name(fn)+ "_graph2D.png"
    plt.savefig(fn_out)
    
################################
#    Draw frame by frame
#################################

def colorize_images(images):

    im, im1, seg, skel = images

    b_seg = seg>0
    b_skel  = skel > 0
    b_skel = morphology.binary_dilation(skel,np.ones((3,3,3)))*1.

    im   = im / im.max()*255
    im1  = im1  / im1.max() * 255 
    b_seg  = b_seg  / b_seg.max() * 255
    b_skel = b_skel / b_skel.max() *255

    clrs = np.array([[1,1,1],[1,0,0],[0,1,0]])
    clr_map = np.zeros((50,3))
    clr_map[:,0] = np.arange(len(clr_map))*5
    clr_map[:,2] = np.arange(len(clr_map),0,-1)*5
    clr_map[0,:] = 0

    d_skel = (49* seg / seg.max()).astype(np.int) 

    b_seg[b_skel>0]=0

    c_im   = im[...,None]*clrs[0]
    c_im1  = im1[...,None]*clrs[0]
    c_seg  = b_seg[...,None]*clrs[1]
    c_skel = b_skel[...,None]*clrs[2]
    c_dist = clr_map[d_skel]

    color_ims = (c_im,c_im1,c_seg,c_skel,c_dist)

    return color_ims

def zxz_frames(color_ims):

    c_im,c_im1,c_seg,c_skel,c_dist = color_ims
        
    frames = []

    for z in range(len(c_im)):

        pnl0 = c_im[z]
        pnl1 = c_im1[z]
        pnl2 = 0.5*c_im1[z] + 0.5*c_seg[z] + 0.5*c_skel[z]
        pnl3 = c_dist[z]

        draw = np.hstack((pnl0,pnl2,pnl3)).astype(np.uint8)
        frames.append(draw)

    pnl0 = c_im.mean(axis=0)
    pnl1 = c_im1.mean(axis=0)
    pnl2 = 0.5*c_im1.mean(axis=0) + \
           0.5*c_seg.mean(axis=0) + \
           0.5*c_skel.mean(axis=0)
    pnl3 = c_dist.mean(axis=0)
    draw = np.hstack((pnl0,pnl2,pnl3)).astype(np.uint8)
    frames.append(draw)

    pnl0 = c_im.max(axis=0)
    pnl1 = c_im1.max(axis=0)
    pnl2 = 0.5*c_im1.max(axis=0) + \
           0.5*c_seg.max(axis=0) + \
           0.5*c_skel.max(axis=0)
    pnl3 = c_dist.max(axis=0)
    draw = np.hstack((pnl0,pnl2,pnl3)).astype(np.uint8)
    frames.append(draw)

    return frames

def draw_panels(im):
    draw = np.vstack([np.hstack(np.max(I,axis=3)) for I in im])
    return draw

def merge_channels(frames, method="flatten"):
    
    if method is "flatten":
        frames_out = [item for sublist in frames for item in sublist]
    elif method is "montage":
        frames_out = []
        frames = np.array(frames).transpose((1,0,2,3,4))
        for fr_time in frames:
            frame = np.vstack([np.hstack([fr_time[0],fr_time[1]]),
                               np.hstack([fr_time[2],fr_time[3]])])       
            frames_out.append(frame)    
    elif method is "blend":        
        frames_out = []
        
        frames = np.array(frames).transpose((1,0,2,3,4))
        for fr_time in frames:
            frame = np.sum(fr_time*1.,axis=0)
            frame[frame>255]=255
            frame = frame.astype(np.uint8)
            frames_out.append(frame) 
    
    return frames_out

def colorize_frames(spin_fr):
    frs_out = []
    colors = [[1,0,0],[0,1,0],[0,0,1],[1,0,1]]
    colors = np.array(colors).astype(np.uint8)

    for c,frs in enumerate(spin_fr):
        
        sub_frames = []
        for fr in frs:
            fr[fr>1] = 1
            fr_out = (fr*255).astype(np.uint8)
            fr_out = fr_out[...,None] * colors[c]
            sub_frames.append(fr_out)
        frs_out.append( sub_frames )
    
    frs_out = np.array(frs_out)
    return frs_out

def render_frames(outname, frames , FPS=30 ):

    size = np.array(frames[0].shape)[[1,0]]
    size = tuple(size.astype(int))

    fcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(outname,fcc, FPS, size )     


    for fr in frames:
        fr = cv2.resize(fr,size)
        out_vid.write(fr[...,::-1])
    out_vid.release()   