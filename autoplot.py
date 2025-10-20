from pymatgen.electronic_structure.core import Spin
import tkinter
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as spop
import os as os
from scipy.interpolate import interp1d
from pymatgen.io.vasp.outputs import Vasprun
import xml.etree.ElementTree as ET
from abipy.abilab import abiopen
import json
#from abipy.electrons.ebands import set_fermie_to_vbm
#from sumo.electronic_structure.bandstructure import BandStructure



def isfloat(s):
    try:
        float(s)
    except:
        return False
    return True

def naneval(s):
    if isfloat(s):
        return eval(s)
    else:
        return np.nan


def listdist(listA,listB):
    assert(len(listA)==len(listB))
    S=0
    for s in range(len(listA)):
        S+=(listA[s]-listB[s])**2
    return np.sqrt(S)

def read_post_bandstructure_CP2K(bandfile):
    filestr=open(bandfile)
    filevec=[]
    Kvec=[]
    Klen=[]
    bind=0
    Ef=-np.inf
    VBM=0
    Egap=np.inf
    for x in filestr:
        xsp=x.split()
        if xsp==[]:
            continue
        if ((xsp[0][0]=="k") or (xsp[1][0]=="P")):
            filevec.append([])
            continue
        if (isfloat(xsp[0])):
            if "occ" in x:
                Ef=max(Ef,naneval(xsp[-1]))
                VBM=naneval(xsp[-1])
            if "vir" in x:
                Egap=min(Egap,naneval(xsp[-1])-VBM)
            filevec[-1].append(naneval(xsp[-1]))
    filevec=np.array(filevec)-Ef
    return {"band structure" : filevec,"direct band gap" : Egap}

with open("results-asr.gw.json") as f:
    data = json.load(f)

# Navigate to bandstructure energies
bs = data["kwargs"]["data"]["bandstructure"]
shape, dtype, flat = bs["e_int_skn"]["__ndarray__"]
energies = np.array(flat, dtype=dtype).reshape(shape)  # (nspin, nk, nbands)

# Use spin=0 (non-magnetic system)
bands = energies[0]  # shape (nk, nbands)

# --- Find VBM ---
# Take the maximum of all energies below 0 eV (or just max of occupied bands if known).
# If no occupations are available, approximate VBM as the maximum energy among all bands <0.
# For safety, let's take the global maximum across all bands as "VBM candidate".
vbm = data["kwargs"]["data"]["vbm_gw"]

print(f"Valence Band Maximum (VBM): {vbm:.3f} eV")

# Shift bands so that VBM = 0
bands_shifted = bands-vbm


path = data["kwargs"]["data"]["bandstructure"]["path"]["kpts"]["__ndarray__"]
shape, dtype, flat = path
kpts = np.array(flat, dtype=dtype).reshape(shape)  # (nk, 3)

special = data["kwargs"]["data"]["bandstructure"]["path"]["special_points"]

M = np.array(special["M"]["__ndarray__"][2])
K = np.array(special["K"]["__ndarray__"][2])

# Find nearest indices
idx_M = np.argmin(np.linalg.norm(kpts - M, axis=1))
idx_K = np.argmin(np.linalg.norm(kpts - K, axis=1))

print("Index of M:", idx_M)
print("Index of K:", idx_K)

# Build a k-axis (dummy uniform for now, unless kpoints available in JSON)
nk = bands.shape[0]
k_axis = np.linspace(0, 1, nk)

basis="TZV2P-MOLOPT"
mode=""
cutoff=""

dictCP2K1=read_post_bandstructure_CP2K(os.path.join(basis,mode,cutoff,"bandstructure_SCF_and_G0W0"))
bandCP2K1,gap1=dictCP2K1["band structure"],dictCP2K1["direct band gap"]


# Plot
mpl.rcParams.update({'font.size': 24})
for ib in range(bands.shape[1]):
    plt.plot(k_axis, bands_shifted[::-1, ib], "b--",lw=10,label="C2DB")
nkCP2K = bandCP2K1.shape[0]
segaxis=np.linspace(0, 1,(nkCP2K-1)//3)
k_axis = np.concatenate((segaxis*(nk-idx_K)/nk,(nk-idx_K)/nk+segaxis*(idx_K-idx_M)/nk,(nk-idx_M)/nk+segaxis*idx_M/nk,[1]))
for ib in range(bandCP2K1.shape[1]):
    plt.plot(k_axis, bandCP2K1[:, ib], "r",lw=10,label="CP2K"+" "+basis)

plt.legend()
plt.ylim(-10,10)


handles, labels = plt.gca().get_legend_handles_labels()
ax=plt.gca()
ax.tick_params(width=2,size=10)
# Remove duplicates while preserving order
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys())
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
fig=plt.gcf()
fig.set_size_inches(figManager.window.winfo_screenwidth()/fig.dpi,figManager.window.winfo_screenheight()/fig.dpi)


figname=str(basis+mode+cutoff).replace("/","")
plt.savefig("plotdir/"+figname)
