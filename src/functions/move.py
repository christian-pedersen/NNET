import numpy as np
#from scipy import optimize
from functions.subfunctions.calc_flow import *
import random

#########################################################################################
### sub function: initvec return transforms the entries of an array into integers     ###
#########################################################################################
def intt(x):
	return (int)(x)
intvec=np.vectorize(intt)

#########################################################################################
### sub function: update positions and radii					      ###
#########################################################################################
def move(position,bonds,blocked,L,nx,dt,eox,eoy,dP,eps,gamma,closed_tube):


	ny=(int)(nx*3.0**0.5/3.0+1)
	dx=3.0**0.5*L*nx;dy=3.0*L*ny

	##### set displacement vectors (the displacement vector has to be calculated before the position is #####
	##### updated, because all displacement depend on all old coordinates) #####
	deltax=np.zeros(len(position))
	deltay=np.zeros(len(position))
	deltar=np.zeros(len(bonds))


	#bonds[:,2][closed_tube] = 0.1#(np.abs(np.cos(time*np.pi*500))+0.2)/1.2

	##### calculate flow #####
	flow=calc_flow(position,bonds,blocked,dx,dy,eox,eoy,dP)
	#print(flow, max(flow))
	feedback=eps*np.abs(flow)/(1.0+np.abs(flow))
	#print(feedback)
	##### loop over all bonds #####
	bond=intvec(bonds[:,:2])

	dxbond=position[bond[:,0],0]+bonds[:,3]*dx-(position[bond[:,1],0]+bonds[:,5]*dx)
	dybond=position[bond[:,0],1]+bonds[:,4]*dy-(position[bond[:,1],1]+bonds[:,6]*dy)
	length=(dxbond**2.0+dybond**2.0)**0.5

	gamma=2.0*gamma/len(bonds)*(np.sum(bonds[:,2]*length)-len(bonds)*L)

	deltar=length/bonds[:,2]**2.0+feedback[:,0]-gamma*length

	factor=(1.0/bonds[:,2]+gamma*bonds[:,2])/length

	for j in range(len(bond)):
		deltax[bond[j,0]]=deltax[bond[j,0]]-factor[j]*dxbond[j]
		deltax[bond[j,1]]=deltax[bond[j,1]]+factor[j]*dxbond[j]
		deltay[bond[j,0]]=deltay[bond[j,0]]-factor[j]*dybond[j]
		deltay[bond[j,1]]=deltay[bond[j,1]]+factor[j]*dybond[j]

	##### update all positions and bonds #####
	position[:,0]=position[:,0]+deltax*dt
	position[:,1]=position[:,1]+deltay*dt
	bonds[:,2]=bonds[:,2]+deltar*dt

	Rij = bonds[:,2]
	Lij = length

	# outfile = open('figs/area.txt', 'a')
	# outfile.write('%f\t %f\n' %(time, sum([2*np.pi*Rij[j]*Lij[j] for j in range(len(Rij))])))
	# outfile.close()

	return position,bonds,Rij, Lij, flow
