import numpy as np
import random as ran

def fctRo(deltaR):
	Ro=np.random.normal(1.0,deltaR,1)[0]
	if Ro<0.1:
		Ro=0.1
	return Ro

#########################################################################################
### sub function: initvec return transforms the entries of an array into integers     ###
#########################################################################################
def intt(x):
	return (int)(x)
intvec=np.vectorize(intt)

#########################################################################################
### set hexagonal points and bonds (the last four describe bonds across the pbc)      ###
#########################################################################################
def make_network(nx,L,deltaR,deltaL,nblocked0):
	
	ny=(int)(nx*3.0**0.5/3.0+1)
	position=np.zeros((4*nx*ny,3))

	dx=3.0**0.5*L*nx;dy=3.0*L*ny
	
	for j in range(nx):
		for jj in range(ny):
			##### positions #####
			periodicx=3.0**0.5*L*j;periodicy=3.0*L*jj
			position[4*(j*ny+jj)+0,0]=periodicx+0.0			#x-position
			position[4*(j*ny+jj)+0,1]=periodicy+0.0			#y-position
			position[4*(j*ny+jj)+1,0]=periodicx-L*np.sin(np.pi/3.0)
			position[4*(j*ny+jj)+1,1]=periodicy+L*np.cos(np.pi/3.0)
			position[4*(j*ny+jj)+2,0]=periodicx-L*np.sin(np.pi/3.0)+L*np.sin(0.0)
			position[4*(j*ny+jj)+2,1]=periodicy+L*np.cos(np.pi/3.0)+L*np.cos(0.0)
			position[4*(j*ny+jj)+3,0]=periodicx-L*np.sin(np.pi/3.0)+L*np.sin(0.0)+L*np.cos(np.pi/6.0)
			position[4*(j*ny+jj)+3,1]=periodicy+L*np.cos(np.pi/3.0)+L*np.cos(0.0)+L*np.sin(np.pi/6.0)

			##### bonds #####
			if j==0 and jj==0:
				bonds=[[0,1,fctRo(deltaR),0.0,0.0,0.0,0.0],[1,2,fctRo(deltaR),0.0,0.0,0.0,0.0],[2,3,fctRo(deltaR),0.0,0.0,0.0,0.0]]
				bonds=np.append(bonds,[[0,4*ny+1,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[3,4*ny+2,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[3,4,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
			elif j<nx-1 and jj<ny-1:
				bonds=np.append(bonds,[[4*(j*ny+jj)+0,4*(j*ny+jj)+1,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+1,4*(j*ny+jj)+2,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+2,4*(j*ny+jj)+3,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+0,4*((j+1)*ny+jj)+1,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+3,4*((j+1)*ny+jj)+2,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+3,4*(j*ny+(jj+1))+0,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
			elif j<nx-1:
				bonds=np.append(bonds,[[4*(j*ny+jj)+0,4*(j*ny+jj)+1,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+1,4*(j*ny+jj)+2,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+2,4*(j*ny+jj)+3,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+0,4*((j+1)*ny+jj)+1,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+3,4*((j+1)*ny+jj)+2,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+3,4*(j*ny+0)+0,fctRo(deltaR),0.0,0.0,0.0,1.0]],axis=0)
			else:
				bonds=np.append(bonds,[[4*(j*ny+jj)+0,4*(j*ny+jj)+1,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+1,4*(j*ny+jj)+2,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+2,4*(j*ny+jj)+3,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+0,4*(jj)+1,fctRo(deltaR),0.0,0.0,1.0,0.0]],axis=0)
				bonds=np.append(bonds,[[4*(j*ny+jj)+3,4*(jj)+2,fctRo(deltaR),0.0,0.0,1.0,0.0]],axis=0)
				if jj<ny-1:
					bonds=np.append(bonds,[[4*(j*ny+jj)+3,4*(j*ny+(jj+1))+0,fctRo(deltaR),0.0,0.0,0.0,0.0]],axis=0)
			if j==nx-1 and jj==ny-1:
				bonds=np.append(bonds,[[4*(j*ny+jj)+3,4*(j*ny+0)+0,fctRo(deltaR),0.0,0.0,0.0,1.0]],axis=0)
	for j in range(len(position)):
		position[j,2]=3	#number of neighbors connected by bonds

	##### sort bond, such that the first index is always smaller than the second
	for j in range(len(bonds)):
		if bonds[j,0]>bonds[j,1]:
			index1=(int)(bonds[j,1]);index2=(int)(bonds[j,0])
			dx1=bonds[j,5];dy1=bonds[j,6]
			dx2=bonds[j,3];dy2=bonds[j,4]
			bonds[j,0]=index1;bonds[j,1]=index2
			bonds[j,3]=dx1;bonds[j,4]=dy1
			bonds[j,5]=dx2;bonds[j,6]=dy2

	##### add random displacement to points #####
	if deltaL>0.0:	
		for j in range(len(position)):
			position[j,0]=position[j,0]+np.random.normal(0.0,deltaL*L,1)
			position[j,1]=position[j,1]+np.random.normal(0.0,deltaL*L,1)

	##### adjust radii to fix total area #####
	areafinal=len(bonds)*L
	x1=position[intvec(bonds[:,0]),0]+bonds[:,3]*dx;x2=position[intvec(bonds[:,1]),0]+bonds[:,5]*dx
	y1=position[intvec(bonds[:,0]),1]+bonds[:,4]*dy;y2=position[intvec(bonds[:,1]),1]+bonds[:,6]*dy
	bondlength=((x1-x2)**2.0+(y1-y2)**2.0)**0.5
	area=np.sum(bondlength*bonds[:,2])
	bonds[:,2]=bonds[:,2]/area*areafinal


	##### make blocked tubes #####
	blocked=np.zeros((len(bonds),1));blocked[:,0]=blocked[:,0]+1.0 #blocked is the list of bond indices, where the entry is set to zero, if the bond is blocked

	checknetwork=0
	newblock=0 #counts how often we tried to set a blocked tube. After 20 failed attempts the loop stops

	while checknetwork<1 and len(np.where(blocked==0)[0])<nblocked0: 
		#blockedout=1.0*blocked #save a configuration of blocked tubes, where still each point in the network can be reached to 'blockedout'

		### block a random tube ###
		index=np.where(blocked==1.0)[0]
		nbreak=(int)(ran.randrange(len(index)-1))
		blocked[index[nbreak]]=0.0

		### delete all blocked tubes from 'bonds' and save it to 'bondscheck'
		block=np.arange(len(bonds))
		block=block[blocked[:,0]<1.0] #blocked is the list of bond indices, where the entry is set to zero, if the bond is blocked
		bondscheck=np.delete(bonds[:],block,0)

		#################################################################################
		### check if every point in the network can be reached from each point        ###
		#################################################################################
		nnmat=np.zeros((len(position),len(position)))

		bond=intvec(bondscheck[:,:2])

		for j in range(len(bondscheck)):
			indexi=bond[j,0]
			indexj=bond[j,1]

			nnmat[indexi,indexj]=-1.0
			nnmat[indexj,indexi]=-1.0
			nnmat[indexi,indexi]=nnmat[indexi,indexi]+1.0
			nnmat[indexj,indexj]=nnmat[indexj,indexj]+1.0

		for k in range(len(nnmat)):
			nnmat[0,k]=0.0
		nnmat[0,0]=1.0

		### if det(nmat)=0, not all points in the network can be reached ###
		if np.abs(np.linalg.det(nnmat))==0.0:
			blocked[index[nbreak]]=1.0 #the tube with index 'nbreak' is unblocked again
			newblock=newblock+1
			if newblock==20:
				checknetwork=1
		else:
			newblock=0
			#index=np.where(blocked<1)[0]
			#np.savetxt(outputfile1,index[:-1])

	return position,bonds,blocked

