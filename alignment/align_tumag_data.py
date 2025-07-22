
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

def to_minimize(p, y, fmodel, fargs, fkwargs, unc):

  model = fmodel(p, *fargs, **fkwargs)

  #res = np.nansum((y-model)**2)*unc
  res = np.nanmean((y-model)**2)*unc

  #print(res, p)

  return res

def to_minimize_corr(p, y, fmodel, fargs, fkwargs):

  model = fmodel(p, *fargs, **fkwargs)

  mask = model+y
  res = - np.abs(pearsonr(y[mask==mask], model[mask==mask])[0])

  return res
 
def _rebin(im,rr=1):

  ny,nx = im.shape

  ry = ny//rr
  rx = nx//rr

  sny = ry*rr
  snx = rx*rr

  x0 = (nx-snx)//2
  y0 = (ny-sny)//2

  return np.mean(im[y0:y0+sny,x0:x0+snx].reshape(ry,rr,rx,rr), axis=(1,3))

def _normalize(im):
  return (im-np.nanmean(im))/np.nanstd(im)

def walign(files, overwrite=False,fovsize=300,istokes = 3):

  ntimes = len(files)
  assert ntimes>1
  npyfiles = []
  totdt = 0.
  totts = 0
  for itime in range(ntimes):
  
    stmp = files[itime].split("/")
    path = "/".join(stmp[:-1])
    file_wofmt = ".".join(stmp[-1].split(".")[:-1])
  
    oname = "%s/%s_waligned.npy" % (path,file_wofmt,)
    npyfiles.append(oname)
  
    if (exists(oname) & (not overwrite)):
      continue
  
    tmp = readfits(files[itime])[1].copy()
    rr = np.max([2,np.mean(tmp.shape[-2:]).astype("i4")//fovsize])
    #print("\n", rr, ":", istokes, "\n")
    import time as tm
    tt0 = tm.time()
    for iw in range(tmp.shape[0]-1):
      print("\ttime=%4i/%4i ; wavelength=%2i/%2i ; reamining ~ %20.5f sec." % (itime,ntimes-1,iw,tmp.shape[0]-2,(ntimes-itime)*(totdt/np.max([totts,1.])),), flush=True,end='\r')


      #isub = tmp[iw+1,istokes,:,:]
      isub = _rebin(_normalize(tmp[iw+1,istokes,:,:]), rr=rr)

      tmpres = []
      for iw2 in range(iw+1):
        #iref = tmp[iw2,istokes,:,:]
        iref = _rebin(_normalize(tmp[iw2,istokes,:,:]), rr=rr)
        minim_args = (iref,rotate_shift, (isub,),{},)  
        tmpres.append(to_minimize_corr([0.,0.,0.], *minim_args))
        #print("\n\n",iw2, tmpres[-1],"\n\n")
      iw2 = np.argmin(tmpres)
      #print(iw2)
      iref = _rebin(_normalize(tmp[iw2,istokes,:,:]), rr=rr)
      minim_args = (iref,rotate_shift, (isub,),{},)  

      tt1 = tm.time()
      #minim_args = (_rebin(tmp[iw,istokes,:,:], rr=rr),rotate_shift, (_rebin(tmp[iw+1,istokes,:,:],rr=rr),),{},)  
      bounds = [(-15./rr,15./rr),(-15./rr,15./rr),(-5,5)]
      tres = minimize(to_minimize_corr, [0.,0.,0.], args=minim_args, bounds=bounds, method="Powell",options={"xtol":0.1,"ftol":0.1})
      #res = minimize(to_minimize_corr, tres.x, args=minim_args, bounds=bounds, method="L-BFGS-B",)
      initial_simplex = np.random.randn(len(tres.x)+1,len(tres.x))
      initial_simplex[:,0:2] /= rr 
      initial_simplex = initial_simplex + tres.x[None,:]
      res = minimize(to_minimize_corr, tres.x, args=minim_args, bounds=bounds, method="Nelder-Mead",options={"fatol":np.abs(tres.fun)*1.e-3, "initial_simplex":initial_simplex})


      success = res.success
      for i in range(res.x.size):
        fact = rr if i<2 else 1.
        db = bounds[i][1] - bounds[i][0]
        t0 = np.abs(bounds[i][0]-res.x[i])/db
        t1 = np.abs(bounds[i][1]-res.x[i])/db
        if ((t1<0.01) | (t0<0.01)):
          success = False
          print("")
          for i in range(res.x.size):
            fact = rr if i<2 else 1.
            print(bounds[i][0]*fact, res.x[i]*fact, bounds[i][1]*fact)
          print("")
          import pdb
          pdb.set_trace()

      if(not success):
        import pdb
        pdb.set_trace()

      #ttmp = tmp.copy()
      shift = res.x * 1.
      shift[0:2] *= rr
      for ij in range(tmp.shape[1]):
        tmp[iw+1,ij,:,:] = minim_args[1](shift, tmp[iw+1,ij,:,:])
      #import pdb
      #pdb.set_trace()
    
      #print("\n"*2, tt2-tt1, tt1-tt0, "\n"*2)
    tt2 = tm.time()
    totdt += (tt2-tt0)
    totts += 1
    np.save(oname, tmp)

  return npyfiles

def talign(files, overwrite=False, fovsize=300, istokes=0):

  import time as tm

  ntimes = len(files)

  totdt = 0.
  totts = 0
  for itime in range(ntimes):
  
    stmp = files[itime].split("/")
    path = "/".join(stmp[:-1])
    file_wofmt = ".".join(stmp[-1].split(".")[:-1])
  
    oname = "%s/%s_taligned.npy" % (path,file_wofmt,)
    sub = np.load(files[itime])
  
    if (exists(oname) & (not overwrite)):
      ref = sub.copy()
      continue
  
    if (itime==0):
      ref = sub.copy()
      np.save(oname, sub)
      continue
  
    tt0 = tm.time()

    print("\ttime=%4i/%4i ; remaining ~ %20.5f sec." % (itime,ntimes-1,(ntimes-1-itime)*(totdt/np.max([totts,1.])),), flush=True,end='\r')
       

    test = np.zeros((sub.shape[0],3))+np.nan
##
    #print("A")
    rr = np.mean(ref.shape[-2:]).astype("i4")//fovsize

    #nas, naf = [i//rr for i in sub.shape[-2:]]
    #bounds = [(-nas//5,nas//5),(-naf//5,naf//5),(-50,50)]
    for iw in range(sub.shape[0]):

      iref = _rebin(_normalize(ref[iw,istokes,:,:]), rr=rr)
      isub = _rebin(_normalize(sub[iw,istokes,:,:]), rr=rr)
      #iref = _rebin((ref[iw,0,:,:]-np.nanmean(ref[iw,0,:,:]))/np.nanstd(ref[iw,0,:,:]), rr=rr)
      #isub = _rebin((sub[iw,0,:,:]-np.nanmean(sub[iw,0,:,:]))/np.nanstd(sub[iw,0,:,:]), rr=rr)

      nas, naf = iref.shape
      bounds = [(-(nas//2),nas//2),(-(naf//2),naf//2),(-90.,90.)]
      minim_args = (iref,rotate_shift, (isub,),{},1.)#/np.nansum(iref*iref*0.01*0.01))  

      #initial_simplex = np.random.rand(len(bounds)+1,len(bounds))
      #for i in range(len(bounds)):
      #  initial_simplex[:,i] = initial_simplex[:,i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

      #res = minimize(to_minimize, [0.,0.,0.], args=minim_args, bounds=bounds, method="Nelder-Mead", options={"xatol":0.1, "fatol":0.1,"initial_simplex":initial_simplex})#,"ftol":1.e-8})
      tres = minimize(to_minimize, [0.,0.,0.], args=minim_args, bounds=bounds, method="Powell", options={"xtol":0.1, "ftol":0.1})#,"ftol":1.e-8})

      #initial_simplex = np.random.randn(len(tres.x)+1,len(tres.x)) + tres.x[None,:]
      initial_simplex = np.random.randn(len(tres.x)+1,len(tres.x))
      initial_simplex[:,0:2] /= rr 
      initial_simplex = initial_simplex + tres.x[None,:]
      res = minimize(to_minimize, tres.x, args=minim_args, bounds=bounds, method="Nelder-Mead", options={"xatol":1.e-6,"fatol":np.abs(tres.fun)*1.e-3,"initial_simplex":initial_simplex, "maxfev":100000, "maxiter":100000})#,"ftol":1.e-8})

      if (res.success):
        #bounds = [(-15,15),(-15,15),(-5,5)]
        test[iw,:] = res.x
      else:
        import pdb
        pdb.set_trace()
      #print(res)
      #import pdb
      #pdb.set_trace()

    test[:,0:2] *= rr  

#    if (np.nanstd(test,0).max()>1):
#    #if (np.abs(np.nanstd(test,0)/np.nanmean(test,0)).max()>1):
#      print("\nWarning! Time alignment per wavelenght shows a deviation >1\n")
#    #  import pdb
#    #  pdb.set_trace()

    #tsub = sub.copy()
    shift = np.nanmedian(test, axis=0)
    for iw in range(sub.shape[0]):
      for ij in range(sub.shape[1]):
        sub[iw,ij,:,:] = minim_args[1](shift, sub[iw,ij,:,:])
    #import pdb
    #pdb.set_trace()
    
    np.save(oname, sub)
    ref = sub.copy()
    tt1 = tm.time()
    totdt += (tt1-tt0)
    totts += 1

  return


