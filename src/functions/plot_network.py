import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm

#########################################################################################
### sub function: initvec return transforms the entries of an array into integers     ###
#########################################################################################
def intt(x):
	return (int)(x)
intvec=np.vectorize(intt)

#########################################################################################
### plot network				   				      ###
#########################################################################################
def plot_network(position,bonds,blocked,L,nx):
	#print(L)
	ny=(int)(nx*3.0**0.5/3.0+1)
	dx=3.0**0.5*L*nx;dy=3.0*L*ny

	##### plot bonds #####
	bond=intvec(bonds[:,:2])
	for j in range(len(bond)):
		x1=position[bond[j,0],0]+bonds[j,3]*dx;x2=position[bond[j,1],0]+bonds[j,5]*dx
		y1=position[bond[j,0],1]+bonds[j,4]*dy;y2=position[bond[j,1],1]+bonds[j,6]*dy
		if blocked[j]==1:
			plt.plot([x1,x2],[y1,y2],'k-',lw=1)
		else:
			plt.plot([x1,x2],[y1,y2],'k--',lw=1)

	##### plot points #####
	for j in range(len(position)):
		plt.plot(position[j,0],position[j,1],'ko',markersize=6)

	##### some settings to make it look nice #####
	plt.xlabel(' ',fontsize=20);plt.ylabel(' ',fontsize=20);plt.tick_params(axis='both', which='major', labelsize=20);
	plt.tight_layout();plt.axes().set_aspect('equal')
	#plt.xlim([-1.2*L,max(dx,dy)-0.8*L]);plt.ylim([-0.2*L,max(dx,dy)+0.2*L])

	plt.show()
	return 0.0
