
from glob import glob as ls

from auxiliars import walign, talign

if (__name__=="__main__"):

  overwrite = False
  path = "/Users/adpa7375/Documents/collaborations/collaboration_sunrise/azaymi/06_SPOT"
  pattern = "06_SPOT_TM_00_Mg1_10_12072024T??????_LV_1.1_v1.1.fits"
  files = sorted(ls("%s/%s" % (path, pattern,)))

  if (len(files)>0):
    wfiles = walign(files, overwrite=overwrite)

  if (len(wfiles)>0):
    talign(wfiles, overwrite=overwrite)






