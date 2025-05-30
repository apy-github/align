
import numpy as np
import matplotlib.pyplot as pl
pl.ion()

from scipy.ndimage import map_coordinates
from os.path import exists
from glob import glob as ls
import time as tm

from scipy.stats import pearsonr

from scipy.optimize import minimize

def readfits(fname, path = './', verbose=False, ext=0, mode="readonly", only_header=False):
  from astropy.io import fits as pf
  from os.path import exists 
  if (len(fname.split("/"))>1):
    tmp = fname.split("/")   
    fname = tmp[-1]          
    path = "/".join(tmp[:-1])
  if (not exists("%s/%s" % (path, fname,))):
    print("\n\tio.readfits :: %s/%s does not exist! (returning None...)" % (path, fname,))
    return None, None        
  if (verbose == True):      
    print("Reading file: %s" % fname)
  hdulist = pf.open("%s/%s" % (path, fname,), mode=mode)
  header = hdulist[ext].header
  if (only_header!=False):   
    hdulist.close()          
    return header, None      
  data = hdulist[ext].data   
  hdulist.close()            
  return header, data    


def check_sequence(data, fnum=1, interval=0.3):
  dims = data.shape
  assert len(dims)==3
  nt,ny,nx = dims
  pl.close(fnum)
  nn = np.max([ny,nx])/4.
  fsize = (ny/nn,nx/nn)
  fg, ax = pl.subplots(nrows=1,ncols=1,clear=1,figsize=fsize,num=fnum,squeeze=False)
  fg.subplots_adjust(hspace=0,wspace=0, left=0,bottom=0,right=1,top=1)
  vmin, vmax = np.nanpercentile(data, [1,99])
  for iti in range(nt):
    ax.flat[0].imshow(data[iti,:,:], interpolation="none", aspect="equal", vmin=vmin,vmax=vmax,cmap="Grays")
    pl.pause(interval)

  return

def rotate_shift(p, im):

  from scipy.ndimage import gaussian_filter, map_coordinates as int2d
  dang = p[2]
  dx = p[0]
  dy = p[1]

  ny, nx = im.shape        
                           
  x2d = np.arange(nx, dtype="f8")[None,:] * np.ones(ny)[:,None]
  y2d = np.arange(ny, dtype="f8")[:,None] * np.ones(nx)[None,:]

  cx = np.nanmean(x2d)   
  cy = np.nanmean(y2d)   
                           
  to_rot = np.vstack([x2d.reshape(1,-1) - cx, y2d.reshape(1,-1) - cy])
  c = np.cos(dang/180.*np.pi)
  s = np.sin(dang/180.*np.pi)
  rot = np.array([c,-s,s,c]).reshape(2,2)
  rotated = np.dot(rot, to_rot) + np.array([cx-dx, cy-dy])[:,None]
                           
  tmp = im*1.              
  tmp[tmp!=tmp] = 0        
  dum = int2d(tmp, rotated[::-1,:]).reshape(ny,nx)

  mask = tmp * 0. + 1.
  rmask = gaussian_filter(int2d(mask, rotated[::-1,:]).reshape(ny,nx),(0.9,0.9))
  rmask[np.abs(rmask-1.)>1.e-1] = np.nan
  return dum * rmask

def to_minimize(p, y, fmodel, fargs, fkwargs):

  model = fmodel(p, *fargs, **fkwargs)

  dy = 0.01 * y
  res = np.nansum((y-model)**2)/np.nansum(dy*dy)

  #print(res, p)

  return res

def to_minimize_corr(p, y, fmodel, fargs, fkwargs):

  model = fmodel(p, *fargs, **fkwargs)

  mask = model+y
  res = - pearsonr(y[mask==mask], model[mask==mask])[0]

  return res

def walign(files, overwrite=False):

  ntimes = len(files)
  assert ntimes>1
  npyfiles = []
  for itime in range(ntimes):
  
    stmp = files[itime].split("/")
    path = "/".join(stmp[:-1])
    file_wofmt = ".".join(stmp[-1].split(".")[:-1])
  
    oname = "%s/%s_waligned.npy" % (path,file_wofmt,)
    npyfiles.append(oname)
  
    if (exists(oname) & (not overwrite)):
      continue
  
    tmp = readfits(files[itime])[1].copy()
    for iw in range(tmp.shape[0]-1):
      print("\ttime=%4i/%4i ; wavelength=%2i/%2i" % (itime,ntimes-1,iw,tmp.shape[0]-2,), flush=True,end='\r')
      minim_args = (tmp[iw,0,:,:],rotate_shift, (tmp[iw+1,0,:,:],),{},)  
      bounds = [(-15,15),(-15,15),(-5,5)]
      res = minimize(to_minimize_corr, [0.,0.,0.], args=minim_args, bounds=bounds, method="Powell",options={"xtol":0.01,"ftol":0.01})
      for ij in range(tmp.shape[1]):
        tmp[iw+1,ij,:,:] = minim_args[1](res.x, tmp[iw+1,ij,:,:])
    
    np.save(oname, tmp)

  return npyfiles

def talign(files, overwrite=False):

  ntimes = len(files)
  for itime in range(ntimes):
  
    stmp = files[itime].split("/")
    path = "/".join(stmp[:-1])
    file_wofmt = ".".join(stmp[-1].split(".")[:-1])
  
    oname = "%s/%s_taligned.npy" % (path,file_wofmt,)
  
    if (exists(oname) & (not overwrite)):
      continue
  
    sub = np.load(files[itime])
    if (itime==0):
      ref = sub.copy()
      np.save(oname, sub)
      continue
  
    print("\ttime=%4i/%4i" % (itime,ntimes-1,), flush=True,end='\r')
  
    test = np.zeros((sub.shape[0],3))+np.nan
    bounds = [(-15,15),(-15,15),(-5,5)]
    for iw in range(sub.shape[0]):
      minim_args = (ref[iw,0,:,:],rotate_shift, (sub[iw,0,:,:],),{},)  
      res = minimize(to_minimize, [0.,0.,0.], args=minim_args, bounds=bounds, method="Powell", options={"xtol":0.01})
      if (res.success):
        test[iw,:] = res.x
  
    shift = np.nanmedian(test, axis=0)
    for iw in range(sub.shape[0]):
      for ij in range(sub.shape[1]):
        sub[iw,ij,:,:] = minim_args[1](shift, sub[iw,ij,:,:])
    
    np.save(oname, sub)
    ref = sub.copy()

  return


