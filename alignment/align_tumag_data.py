
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

def uto_minimize(up, u2d, y, fmodel, fargs, fkwargs, unc):

  dp = up * u2d[:,0] + u2d[:,1]
  model = fmodel(dp, *fargs, **fkwargs)

  #res = np.nansum((y-model)**2)*unc
  res = np.nanmean((y-model)**2)*unc

  #print(res, p)

  return res

def to_minimize_corr(p, y, fmodel, fargs, fkwargs):

  model = fmodel(p, *fargs, **fkwargs)

  mask = model+y
  res = - np.abs(pearsonr(y[mask==mask], model[mask==mask])[0])

  return res

def uto_minimize_corr(up, u2d, y, fmodel, fargs, fkwargs):

  dp = up * u2d[:,0] + u2d[:,1]
  model = fmodel(dp, *fargs, **fkwargs)

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

def twalign(ifile, overwrite, fovsize, istokes, writeoffsets):

  walign([ifile,], overwrite=overwrite,fovsize=fovsize,istokes = istokes, writeoffsets=writeoffsets)

  return

def pwalign(files, overwrite=False,fovsize=300,istokes = 3, writeoffsets=False, nthreads=1):

  from multiprocessing import Process, Queue

  def worker(input, output):
    for func, args, kwargs in iter(input.get, 'STOP'):
      result = func(*args, **kwargs)
      output.put(result)

  NUMBER_OF_PROCESSES = nthreads
  TASKS = [(twalign, (i, overwrite, fovsize, istokes, writeoffsets), {}) for i in files]
  
  # Create queues
  task_queue = Queue()
  done_queue = Queue()
  
  # Submit tasks
  for task in TASKS:
      task_queue.put(task)
  
  # Start worker processes
  for i in range(NUMBER_OF_PROCESSES):
      Process(target=worker, args=(task_queue, done_queue)).start()
  
  # Get and print results
  for i in range(len(TASKS)):
      done_queue.get()
      #print('\t', done_queue.get())
  
  # Tell child processes to stop
  for i in range(NUMBER_OF_PROCESSES):
      task_queue.put('STOP')

  return

def walign(files, overwrite=False,fovsize=300,istokes = 3, writeoffsets=False):

  from scipy import optimize
  import time as tm
  from copy import deepcopy as cp

  ntimes = len(files)
  assert ntimes>0
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

    tt0 = tm.time()

    offsets = np.zeros((tmp.shape[0],3),dtype="f8") + np.nan

    print("\n%s\n" % (files[itime]))

    for iw in range(tmp.shape[0]-1):
#      print("\ttime=%4i/%4i ; wavelength=%2i/%2i ; remaining ~ %20.5f sec." % (itime,ntimes-1,iw,tmp.shape[0]-2,(ntimes-itime)*(totdt/np.max([totts,1.])),))#, flush=True,end='\r')

      isub = _rebin(_normalize(tmp[iw+1,istokes,:,:]), rr=rr)

      tmpres = []
      for iw2 in range(iw+1):
        iref = _rebin(_normalize(tmp[iw2,istokes,:,:]), rr=rr)
        minim_args = (iref,rotate_shift, (isub,),{},)  
        tmpres.append(to_minimize_corr([0.,0.,0.], *minim_args))

      iw2 = np.argmin(tmpres)
      iref = _rebin(_normalize(tmp[iw2,istokes,:,:]), rr=rr)
      minim_args = (iref,rotate_shift, (isub,),{},)  

      tt1 = tm.time()
      bounds = [(-25./rr,25./rr),(-25./rr,25./rr),(-10.,10.)]
      eps = [1.e-3 * (i[1]-i[0]) for i in bounds]

      # Initial guess:
      u2d = np.hstack([np.diff(np.array(bounds), axis=1), np.min(np.array(bounds), axis=1).reshape(-1,1)])
      ubounds = [(0.,1.),] * len(bounds)
      uminim_args = (u2d, iref,rotate_shift, (isub,),{},)  

      tbres2 = minimize(uto_minimize_corr, [0.5,]*len(ubounds), args=uminim_args, bounds=ubounds, method="Powell")

      # Reduce bounds and use Nelder-Mead:
      nbounds = []
      x0g = tbres2.x*u2d[:,0]+u2d[:,1]
      for j, i in enumerate(bounds):
        dbound = (i[1] - i[0])/5
        nbounds.append((x0g[j]-dbound, x0g[j]+dbound))

      # Ubounds remains the same (0,1) but transformation from adimensional to dimensional changes:
      nu2d = np.hstack([np.diff(np.array(nbounds), axis=1), np.min(np.array(nbounds), axis=1).reshape(-1,1)])
      ubounds = [(0.,1.),] * len(nbounds)
      uminim_args = (nu2d, iref,rotate_shift, (isub,),{},)  

      # Final refinement:
      res = minimize(uto_minimize_corr, tbres2.x, args=uminim_args, bounds=ubounds, method="Nelder-Mead",options={"xatol":1.e-5, "maxiter":10000, "maxfev":10000})

      res.x = res.x*nu2d[:,0]+nu2d[:,1]


      # Check boundaries:
      success = res.success
      for i in range(res.x.size):
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
      #

      if(not success):
        print(res)
        import pdb
        pdb.set_trace()

      shift = res.x * 1.
      shift[0:2] *= rr
      for ij in range(tmp.shape[1]):
        tmp[iw+1,ij,:,:] = minim_args[1](shift, tmp[iw+1,ij,:,:])

      offsets[iw+1,:] = shift.copy()

    if (writeoffsets==True):
      optoname = "%s/%s_woffsets.npy" % (path,file_wofmt,)
      np.save(optoname, offsets)

    tt2 = tm.time()
    totdt += (tt2-tt0)
    print("\n%s [%.4f sec.] \n" % (files[itime], tt2-tt0,))
    totts += 1
    np.save(oname, tmp)

  return npyfiles
#
#
#
# Time alignment:
#
def talign(files, overwrite=False, fovsize=300, istokes=0, writeoffsets=False):

  from scipy import optimize

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
    #
    rr = np.mean(ref.shape[-2:]).astype("i4")//fovsize

    for iw in range(sub.shape[0]):

      iref = _rebin(_normalize(ref[iw,istokes,:,:]), rr=rr)
      isub = _rebin(_normalize(sub[iw,istokes,:,:]), rr=rr)

      nas, naf = iref.shape
      bounds = [(-(nas//2),nas//2),(-(naf//2),naf//2),(-90.,90.)]
#
#
      # Initial guess:
      u2d = np.hstack([np.diff(np.array(bounds), axis=1), np.min(np.array(bounds), axis=1).reshape(-1,1)])
      ubounds = [(0.,1.),] * len(bounds)
      uminim_args = (u2d, iref,rotate_shift, (isub,),{},1.)  

      tbres2 = minimize(uto_minimize, [0.5,]*len(ubounds), args=uminim_args, bounds=ubounds, method="Powell")

      # Reduce bounds and use Nelder-Mead:
      nbounds = []
      x0g = tbres2.x*u2d[:,0]+u2d[:,1]
      for j, i in enumerate(bounds):
        dbound = (i[1] - i[0])/5
        nbounds.append((x0g[j]-dbound, x0g[j]+dbound))

      # Ubounds remains the same (0,1) but transformation from adimensional to dimensional changes:
      nu2d = np.hstack([np.diff(np.array(nbounds), axis=1), np.min(np.array(nbounds), axis=1).reshape(-1,1)])
      ubounds = [(0.,1.),] * len(nbounds)
      uminim_args = (nu2d, iref,rotate_shift, (isub,),{},1.,)  

      # Final refinement:
      res = minimize(uto_minimize, tbres2.x, args=uminim_args, bounds=ubounds, method="Nelder-Mead",options={"xatol":1.e-5, "maxiter":10000, "maxfev":10000})

      res.x = res.x*nu2d[:,0]+nu2d[:,1]


      if (res.success):
        test[iw,:] = res.x
      else:
        import pdb
        pdb.set_trace()
    #
    test[:,0:2] *= rr  

#    if (np.nanstd(test,0).max()>1):
#    #if (np.abs(np.nanstd(test,0)/np.nanmean(test,0)).max()>1):
#      print("\nWarning! Time alignment per wavelenght shows a deviation >1\n")
#    #  import pdb
#    #  pdb.set_trace()

    shift = np.nanmedian(test, axis=0)
    for iw in range(sub.shape[0]):
      for ij in range(sub.shape[1]):
        sub[iw,ij,:,:] = minim_args[1](shift, sub[iw,ij,:,:])
    
    np.save(oname, sub)
    ref = sub.copy()

    if (writeoffsets==True):
      optoname = "%s/%s_woffsets.npy" % (path,file_wofmt,)
      np.save(optoname, offsets)

    tt1 = tm.time()

    import pdb
    pdb.set_trace()

    totdt += (tt1-tt0)
    totts += 1

  return





def ptalign1_step(iw, rr, fmap, wiref, wisub):

  from scipy import optimize

  iref = _rebin(_normalize(wiref), rr=rr)
  isub = _rebin(_normalize(wisub), rr=rr)

  nas, naf = iref.shape
  bounds = [(-(nas//2),nas//2),(-(naf//2),naf//2),(-90.,90.)]
  #eps = [1.e-3 * (i[1]-i[0]) for i in bounds]

  minim_args = (iref, fmap, (isub,),{},1.)

  tbres2 = optimize.direct(to_minimize, bounds, args=minim_args, maxfun=300000, maxiter=100000, vol_tol=1.e-14)
  #tbres3 = optimize.minimize(to_minimize, [0,0,0,], args=minim_args, bounds=bounds, options={"eps":eps,})
  if (not tbres2.success):
    print("\n"*10)
    print(tbres2)
    print("\n"*10)
    import pdb
    pdb.set_trace()

  nbounds = []
  for j, i in enumerate(bounds):
    nbounds.append((i[0]/5. + tbres2.x[j],i[1]/5. + tbres2.x[j]))
#  initial_simplex = np.random.rand(len(tbres2.x)+1,len(tbres2.x))
#  for i in range(len(nbounds)):
#    initial_simplex[:,i] = initial_simplex[:,i] * (nbounds[i][1] - nbounds[i][0]) + nbounds[i][0]
  res = optimize.minimize(to_minimize, tbres2.x, args=minim_args, bounds=nbounds, method="Powell")
#
  if (res.success):
    fres = res.x * 1.
  else:
    print("\n"*10)
    print(res)
    print("\n"*10)
    import pdb
    pdb.set_trace()
  #
  fres[0:2] *= rr  
  #
  return iw, fres
#
def ptalign1(ref, sub, fovsize, istokes, fmap, nthreads):

  from scipy import optimize
  from multiprocessing import Process, Queue

  def worker(input, output):
      for func, args, kwargs in iter(input.get, 'STOP'):
          result = func(*args, **kwargs)
          output.put(result)

#  tt0 = tm.time()

  test = np.zeros((sub.shape[0],3))+np.nan
  #
  rr = np.mean(ref.shape[-2:]).astype("i4")//fovsize

#  tt0a = tm.time()
  NUMBER_OF_PROCESSES = nthreads
  TASKS = [(ptalign1_step, (i, rr, fmap, ref[i,istokes,:,:].copy() \
      , sub[i,istokes,:,:].copy()), {}) for i in range(sub.shape[0])]

  # Create queues
  task_queue = Queue()
  done_queue = Queue()
  
  # Submit tasks
  for task in TASKS:
      task_queue.put(task)
  
  # Start worker processes
  for i in range(NUMBER_OF_PROCESSES):
      Process(target=worker, args=(task_queue, done_queue)).start()
  
  # Get and print results
  res = np.zeros((len(TASKS), 2), )
  for i in range(len(TASKS)):
      tmp = done_queue.get()
      test[np.int32(tmp[0]), :] = tmp[1].copy()
  
  # Tell child processes to stop
  for i in range(NUMBER_OF_PROCESSES):
      task_queue.put('STOP')

#  tt1a = tm.time()
#  ctest = test.copy()
#
#
#
#  tt2a = tm.time()
#
#  for iw in range(sub.shape[0]):
#
#    ref[iw,istokes,:,:]
#    sub[iw,istokes,:,:]
#
#    tmp = ptalign1_step(iw, rr, ref[iw,istokes,:,:], sub[iw,istokes,:,:])
#    test[np.int32(tmp[0]), :] = tmp[1].copy()
#
#
#  tt3a = tm.time()
#
#  tt1 = tm.time()
#
#  print("\n"*5, tt3a-tt2a, tt1a-tt0a, ":", "\n"*5)
#
#  import pdb
#  pdb.set_trace()

  return test

#::    sub = ptapply_shift(shift, rotate_shift, sub, nthreads)
#::
def ptapply_shift_step(ij, shift, fmap, im):
  res = fmap(shift, im)
  return ij, res
#::    sub[ij,:,:] = fmap(shift, sub[ij,:,:])
def ptapply_shift(offsets, fmap, sub, nthreads):

  from multiprocessing import Process, Queue

  def worker(input, output):
      for func, args, kwargs in iter(input.get, 'STOP'):
          result = func(*args, **kwargs)
          output.put(result)

  dims = sub.shape
  nn = np.prod(dims[0:-2])
  sub = sub.reshape(nn, dims[-2], dims[-1])

  NUMBER_OF_PROCESSES = nthreads
  TASKS = [(ptapply_shift_step, (i, offsets, fmap \
      , sub[i,:,:].copy()), {}) for i in range(sub.shape[0])]

  # Create queues
  task_queue = Queue()
  done_queue = Queue()
  
  # Submit tasks
  for task in TASKS:
      task_queue.put(task)
  
  # Start worker processes
  for i in range(NUMBER_OF_PROCESSES):
      Process(target=worker, args=(task_queue, done_queue)).start()
  
  # Get and print results
  res = np.zeros((len(TASKS), 2), )
  for i in range(len(TASKS)):
      tmp = done_queue.get()
      sub[np.int32(tmp[0]), :, :] = tmp[1].copy()
  
  # Tell child processes to stop
  for i in range(NUMBER_OF_PROCESSES):
      task_queue.put('STOP')

  return sub.reshape(dims)
 

def ptalign(files, overwrite=False, fovsize=300, istokes=0, writeoffsets=False, nthreads=1):

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

    test = ptalign1(ref, sub, fovsize, istokes, rotate_shift, nthreads)

#    if (np.nanstd(test,0).max()>1):
#    #if (np.abs(np.nanstd(test,0)/np.nanmean(test,0)).max()>1):
#      print("\nWarning! Time alignment per wavelenght shows a deviation >1\n")
#    #  import pdb
#    #  pdb.set_trace()

    shift = np.nanmedian(test, axis=0)

    sub = ptapply_shift(shift, rotate_shift, sub, nthreads)

    np.save(oname, sub)
    ref = sub.copy()

    if (writeoffsets==True):
      optoname = "%s/%s_toffsets.npy" % (path,file_wofmt,)
      np.save(optoname, shift)

    tt1 = tm.time()

    totdt += (tt1-tt0)
    totts += 1

  return

















































def _get_time(fstr):

  fname = fstr.split("/")[-1]
  for i in fname.split("_"):
    if ("2024T" in i):
      break

  date, time = i.split("T")

  dd = np.int32(date[0:2])
  mm = np.int32(date[2:4])
  yy = np.int32(date[4:8])
#
  hour = np.int32(time[0:2])
  mins = np.int32(time[2:4])
  secs = np.int32(time[4:6])

  ftime = ((hour * 60.) + mins) * 60. + secs

  return yy, mm, dd, ftime












def pcam_align(ref_files, sub_files, overwrite=False, fovsize=300, istokes=0, writeoffsets=False, nthreads=16):

  from multiprocessing import Process, Queue

  def worker(input, output):
    for func, args, kwargs in iter(input.get, 'STOP'):
      result = func(*args, **kwargs)
      output.put(result)

  NUMBER_OF_PROCESSES = nthreads
  TASKS = [(tcalign, (ref_files, i, overwrite, fovsize, istokes, writeoffsets), {}) for i in sub_files]
  
  # Create queues
  task_queue = Queue()
  done_queue = Queue()
  
  # Submit tasks
  for task in TASKS:
      task_queue.put(task)
  
  # Start worker processes
  for i in range(NUMBER_OF_PROCESSES):
      Process(target=worker, args=(task_queue, done_queue)).start()
  
#  # Get and print results
#  for i in range(len(TASKS)):
#      print('\t', done_queue.get())
  
  # Tell child processes to stop
  for i in range(NUMBER_OF_PROCESSES):
      task_queue.put('STOP')

  return


def tcalign(ref_files, ifile, overwrite, fovsize, istokes, writeoffsets):

  cam_align(ref_files, [ifile,], overwrite=overwrite,fovsize=fovsize,istokes = istokes, writeoffsets=writeoffsets)

  return


def cam_align(ref_files, sub_files, overwrite=False, fovsize=300, istokes=0, writeoffsets=False):

  from scipy import optimize

  ref_ntimes = len(ref_files)
  sub_ntimes = len(sub_files)

  totdt = 0.
  totts = 0
  for sub_itime in range(sub_ntimes):
  
    isub_file = sub_files[sub_itime]

 
    stmp = isub_file.split("/")
    path = "/".join(stmp[:-1])
    file_wofmt = ".".join(stmp[-1].split(".")[:-1])
  
    oname = "%s/%s_caligned.npy" % (path,file_wofmt,)
  
    if (exists(oname) & (not overwrite)):
      continue
  
    tt0 = tm.time()
    print("\ttime=%4i/%4i ; remaining ~ %20.5f sec." % (sub_itime,sub_ntimes-1,(sub_ntimes-1-sub_itime)*(totdt/np.max([totts,1.])),), flush=True,end='\r')

    isub_yy, isub_mm, isub_dd, isub_time = _get_time(isub_file)

    time_dists = np.zeros((ref_ntimes,),dtype="f8") + np.nan
    for ref_itime in range(ref_ntimes):
      iref_file = ref_files[ref_itime]
      iref_yy, iref_mm, iref_dd, iref_time = _get_time(iref_file)
      if (isub_yy==iref_yy):
        if (isub_mm==iref_mm):
          if (isub_dd==iref_dd):
            time_dists[ref_itime] = iref_time - isub_time

    wwb = time_dists == time_dists
    stime_dists = time_dists[wwb]
    nstime_dists = stime_dists[stime_dists<0]
    pstime_dists = stime_dists[stime_dists>0]

    if (len(nstime_dists)>0):
      dt = nstime_dists[np.argmax(nstime_dists)]
      iref_file = np.array(ref_files)[wwb][np.nanargmin(np.abs(stime_dists-nstime_dists[np.argmax(nstime_dists)]))]
    else:
      #iref_file = np.array(ref_files)[wwb][np.nanargmin(np.abs(stime_dists-pstime_dists[np.argmin(pstime_dists)]))]
      raise Exception("I do not find any previous (in time) reference scan!")

    refim = np.load(iref_file)
    subim = np.load(isub_file)

    srefim = refim[:,istokes,:,:]
    ssubim = subim[:,istokes,:,:]

    #
    # To improve, do not assume overall alignment:

    nwr, nyr, nxr = srefim.shape
    nws, nys, nxs = ssubim.shape

    x0 = 0
    y0 = 0

    x1 = nxr if nxr<nxs else nxs
    y1 = nyr if nyr<nys else nys

    from scipy.stats import pearsonr
    pmax = 0.
    for irw in range(nwr):
      tmprefim = srefim[irw,y0:y1,x0:x1].copy()
      for isw in range(nws):
        tmpsubim = ssubim[isw,y0:y1,x0:x1].copy()

        mask = tmprefim + tmpsubim

        pcc = pearsonr(tmprefim[mask==mask], tmpsubim[mask==mask])[0]
        if (pcc>pmax):
          pmax = pcc * 1.
          jrw = irw
          jsw = isw


    rr = np.max([1, np.mean(refim.shape[-2:]).astype("i4")//fovsize])

    iref = _rebin(_normalize(refim[irw,istokes,y0:y1,x0:x1]), rr=rr)
    isub = _rebin(_normalize(subim[isw,istokes,y0:y1,x0:x1]), rr=rr)

    nas, naf = iref.shape
    bounds = [(-(nas//5),nas//5),(-(naf//5),naf//5),(-30.,30.)]
    minim_args = (iref,rotate_shift, (isub,),{},1.)#/np.nansum(iref*iref*0.01*0.01))  

    tbres2 = optimize.direct(to_minimize, bounds, args=minim_args, maxfun=300000, maxiter=100000, vol_tol=1.e-12)

    nbounds = []
    for j, i in enumerate(bounds):
      nbounds.append((i[0]/5. + tbres2.x[j],i[1]/5. + tbres2.x[j]))
    initial_simplex = np.random.rand(len(tbres2.x)+1,len(tbres2.x))
    for i in range(len(nbounds)):
      initial_simplex[:,i] = initial_simplex[:,i] * (nbounds[i][1] - nbounds[i][0]) + nbounds[i][0]
    res = minimize(to_minimize, tbres2.x, args=minim_args, bounds=nbounds, method="Nelder-Mead",options={"xatol":1.e-3, "initial_simplex":initial_simplex})

    #from userlib import graphics
    #graphics.blink([refim[irw,istokes,y0:y1,x0:x1], subim[isw,istokes,y0:y1,x0:x1]])

    ##graphics.blink([iref, isub])

    #tt = minim_args[1](res.x, isub)

    #graphics.blink([iref, tt])

    corr = res.x * 1.
    corr[0:2] *= rr

    #
    # Apply correction:
    #
#    csubim = subim.copy()
    snw, sns, _, _ = subim.shape
    for inw in range(snw):
      for ins in range(sns):
        subim[inw,ins,:,:] = minim_args[1](corr, subim[inw,ins,:,:])

#    from userlib import graphics
#    graphics.blink([refim[irw,istokes,y0:y1,x0:x1], subim[isw,istokes,y0:y1,x0:x1]])
#
#    import pdb
#    pdb.set_trace()
    np.save(oname, subim)

    if (writeoffsets==True):
      optoname = "%s/%s_coffsets.npy" % (path,file_wofmt,)
      np.save(optoname, corr)

    tt1 = tm.time()
    totdt += (tt1-tt0)
    totts += 1

  return
