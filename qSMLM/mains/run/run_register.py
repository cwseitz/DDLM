import tifffile
from pystackreg import StackReg
from pystackreg.util import to_uint16
from skimage.io import imsave
import json

prefixes = [
'230726_Hela_J549_1pM_1hr_j646_10pM_overnight_5mW_20mW-G-4'
]

with open('run_register.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    sr = StackReg(StackReg.RIGID_BODY)
    path = config['datapath']+prefix
    stack = tifffile.imread(path+'.tif')
    out = sr.register_transform_stack(stack, reference='first')
    path = config['datapath']+prefix
    imsave(path+'-regi.tif',to_uint16(out))
