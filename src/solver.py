import numpy as np
import sys, getopt
from functions.functions.make_network import *
from functions.functions.plot_network import *
from functions.functions.replot_network import *
from functions.functions.move import *


def intt(x):
	return (int)(x)
intvec=np.vectorize(intt)


class Solver():
    def __init__(self, inputfile='input.txt', close_tube=198, optimize_tube=14):

        data = np.loadtxt(inputfile)
        self.nx=(int)(data[0])	#nx

        self.L=data[1] #initial bond length (in units of Ro)
        self.deltaL=data[2] 		#variation bond length (in units of Ro)
        self.deltaR=data[3] 		#variation radius (in units of Ro)

        ##### random seed ####
        self.seed=(int)(data[4])

        if self.seed>=0:
        	print('WARNING: fixed random seed')
        else:
        	self.seed=(int)(100000000*np.random.random_sample(1))

        np.random.seed(self.seed)
        data[4]=self.seed		#random seed

        ##### parameter for dynamics ####
        self.nstep=(int)(data[5]) 	#total number of simulation steps
        self.nout=(int)(data[6]) 	#steps to save result
        self.dt=data[7] 		#time step

        ##### parameter for flow ####
        self.eox=data[8] 		#x-direction (will be normalized)
        self.eoy=data[9] 		#y-direction (will be normalized)
        self.dP=data[10] 		#pressure difference
        self.eps=data[11] 		#feedback prefactor

        ##### parameter for blocked tubes ####
        self.nblocked0=data[12]	#number of tubes that should be blocked

        ##### area constraint ####
        self.gamma=data[14]		#constraint area



        #########################################################################################
        ### set hexagonal points and bonds                                                    ###
        #########################################################################################
        position, bonds, blocked=make_network(self.nx,self.L,self.deltaR,self.deltaL,self.nblocked0)
        data[13]=len(np.where(blocked==0)[0])

        self.close_tube = close_tube
        self.optimize_tube = optimize_tube

        self.init_bonds = bonds

        self.init_position = position
        self.init_blocked = blocked


    def set_up(self):
        self.no = 0
        position, bonds, blocked=make_network(self.nx,self.L,self.deltaR,self.deltaL,self.nblocked0)
        self.number_of_tubes = len(bonds[:,2])
        self.bonds = bonds
        print(self.bonds)
        self.position = position
        self.blocked = blocked
        self.init_radius = self.bonds[:,2]
        self.init_q = 0.5

        return self.number_of_tubes, self.init_radius, self.init_q, self.close_tube, self.optimize_tube

    def evolve(self, action, plot=True):
        position, bonds, Rij, Lij, flow=move(self.position,self.bonds,self.blocked,self.L,self.nx,self.dt,self.eox,self.eoy,self.dP,action,self.gamma, self.close_tube)
        self.position, self.bonds = position, bonds
        self.flow = flow
        self.radius, self.length = Rij, Lij
        self.q_measure = float(self.flow[self.optimize_tube])#2*np.pi*Rij[self.optimize_tube]*Lij[self.optimize_tube]

        return self.radius, self.length, self.q_measure, self.flow

    def plot(self, folder, eps, type):
        pause = 0
        replot_network(self.position,self.bonds,self.blocked,self.L,self.nx,pause,self.eox,self.eoy,self.dP,flowmax=1., no=self.no, folder=folder, eps=eps, type=type)
        replot_network(self.position,self.bonds,self.blocked,self.L,self.nx,pause,self.eox,self.eoy,self.dP,flowmax=1., no=self.no, folder=folder, eps=eps, type=type)
        if type == 1:
            self.no +=1

