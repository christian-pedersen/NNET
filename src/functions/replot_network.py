import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
from functions.subfunctions.calc_flow import *

#########################################################################################
### sub function: initvec return transforms the entries of an array into integers     ###
#########################################################################################
def intt(x):
	return (int)(x)
intvec=np.vectorize(intt)

#########################################################################################
### plot network				   				      ###
#########################################################################################
def replot_network(position,bonds,blocked,L,nx,pause,eox,eoy,dP,flowmax, no, folder, eps, type):
	if type == 0:
		fig=plt.figure(figsize=(38.0/2.54,28/2.54))
		ax = fig.add_subplot(111)

		ny=(int)(nx*3.0**0.5/3.0+1)
		dx=3.0**0.5*L*nx;dy=3.0*L*ny

		plt.ion()	#Turn the interactive mode on (to show a sequence of plots)
		#plt.figure()

		##### plot bonds #####
		bond=intvec(bonds[:,:2])

		flow=np.abs(calc_flow(position,bonds,blocked,dx,dy,eox,eoy,dP))
		flow=calc_flow(position,bonds,blocked,dx,dy,eox,eoy,dP)
		#outfile = open('figs/flow.txt', 'a')
		#print(flow[15], flow[16], flow[21], flow[22], sum(flow[:]))
		#outfile.write('%.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\n'%(time, flow[15], flow[16], flow[21], flow[22], sum(flow[:])))
		#outfile.close()
		cmap = plt.get_cmap('coolwarm')
		
		#flowmax=np.max(flow)
		#flowmax=1000.0
		color_flow = flow + abs(min(flow))
		color_flow /= max(color_flow)

		for j in range(len(bond)):
			color = cmap(color_flow[j])#cmap(1-flow[j]/flowmax)

			#print(color)

			x1=position[bond[j,0],0]+bonds[j,3]*dx;x2=position[bond[j,1],0]+bonds[j,5]*dx
			y1=position[bond[j,0],1]+bonds[j,4]*dy;y2=position[bond[j,1],1]+bonds[j,6]*dy
			#print(bond[j,0], bond[j,1], flow[j])
			if blocked[j]==1:
				p1=plt.plot([x1,x2],[y1,y2],'-',c=color[0],lw=3)
				plt.text((x1+x2)/2,(y1+y2)/2 ,'%g, q=%.3f'%(j, flow[j]))
				plt.text(x1, y1, '%g'%bond[j,0])
				plt.text(x2, y2, '%g'%bond[j,1])
			else:
				p1=plt.plot([x1,x2],[y1,y2],'k:',lw=3)
				plt.text((x1+x2)/2,(y1+y2)/2 + 3,'7h')
				plt.text(x1, y1, '%g'%bond[j,0])
				plt.text(x2, y2, '%g'%bond[j,1])

		##### plot points #####
		for j in range(len(position)):
			plt.plot(position[j,0],position[j,1],'ko',markersize=6)

		##### some settings to make it look nice #####

		sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1.5, vmax=1.5))
		#sm._A = bond

		plt.colorbar(sm)

		plt.xlabel(' ',fontsize=10);plt.ylabel(' ',fontsize=10);plt.tick_params(axis='both', which='major', labelsize=10);
		#plt.tight_layout();
		#plt.colorbar()
		#ax.set_aspect('equal')
		#plt.xlim([-1.2*L,max(dx,dy)-0.8*L]);plt.ylim([-0.2*L,max(dx,dy)+0.2*L])
		plt.savefig(folder+'flow_%04d.png' %no)
		#plt.show();plt.pause(pause);plt.gcf().clear()
		#plt.show();plt.pause(0.1);
		plt.gcf().clear()
		plt.ioff()
		plt.close()
	else:
		fig=plt.figure(figsize=(28.0/2.54,28/2.54))
		ax = fig.add_subplot(111)

		ny=(int)(nx*3.0**0.5/3.0+1)
		dx=3.0**0.5*L*nx;dy=3.0*L*ny

		plt.ion()	#Turn the interactive mode on (to show a sequence of plots)
		#plt.figure()

		##### plot bonds #####
		bond=intvec(bonds[:,:2])

		flow=np.abs(calc_flow(position,bonds,blocked,dx,dy,eox,eoy,dP))
		#outfile = open('figs/flow.txt', 'a')
		#print(flow[15], flow[16], flow[21], flow[22], sum(flow[:]))
		#outfile.write('%.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\n'%(time, flow[15], flow[16], flow[21], flow[22], sum(flow[:])))
		#outfile.close()
		cmap = plt.get_cmap('coolwarm')
		#flowmax=np.max(flow)
		#flowmax=1000.0
		#print(eps)
		for j in range(len(bond)):
			color = cmap(eps[j]/100)#cmap(1-flow[j]/flowmax)

			x1=position[bond[j,0],0]+bonds[j,3]*dx;x2=position[bond[j,1],0]+bonds[j,5]*dx
			y1=position[bond[j,0],1]+bonds[j,4]*dy;y2=position[bond[j,1],1]+bonds[j,6]*dy
			if blocked[j]==1:
				plt.plot([x1,x2],[y1,y2],'-',c=color[0],lw=3)
				plt.text((x1+x2)/2,(y1+y2)/2 ,'%g, eps=%.3f'%(j,eps[j]))
			else:
				plt.plot([x1,x2],[y1,y2],'k:',lw=3)
				plt.text((x1+x2)/2,(y1+y2)/2 + 3,'7h')

		##### plot points #####
		for j in range(len(position)):
			plt.plot(position[j,0],position[j,1],'ko',markersize=6)

		##### some settings to make it look nice #####
		plt.xlabel(' ',fontsize=10);plt.ylabel(' ',fontsize=10);plt.tick_params(axis='both', which='major', labelsize=10);
		plt.tight_layout();
		#plt.colorbar()
		#ax.set_aspect('equal')
		#plt.xlim([-1.2*L,max(dx,dy)-0.8*L]);plt.ylim([-0.2*L,max(dx,dy)+0.2*L])
		plt.savefig(folder+'eps_%04d.png' %no)
		#plt.show();plt.pause(pause);plt.gcf().clear()
		#plt.show();plt.pause(0.1);
		plt.gcf().clear()
		plt.ioff()
		plt.close()
	#fig.close()
	return 0.0
