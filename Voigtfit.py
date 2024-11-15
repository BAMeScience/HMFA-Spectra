# -*- coding: utf-8 -*-

import os
from random import random
from re import S

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from scipy import interpolate
from scipy.signal import find_peaks
import pandas as pd



#assert pyro.__version__.startswith('1.8.0')



class Voigtfit():
    def __init__(self,wl,Data,Peak_threshold = 0.15,GPU=0,max_itr=200):
        assert Peak_threshold>=0.15 , 'threshold must be greater than 0.15 to avoid noise fitting'
        self.wl = wl
        self.Data = Data
        self.threshold = Peak_threshold
        self.max_epoch=max_itr
        torch.cuda.set_device(GPU)



    def FitData(self):
            
        Data_df = self.Data
        Data_array= np.array(Data_df)

        def voigt_vec(x,beta,alpha,pos,gamma):
            print(beta.shape)
            voigt_dist = beta[:,None]*alpha[:,None] *np.exp(-4*np.log(2)*(x-pos[:,None])**2 / (gamma[:,None]**2)) + ((1-beta[:,None])*alpha[:,None]*gamma[:,None]**2 / ((x-pos[:,None])**2 + gamma[:,None]**2))
            return voigt_dist.sum(axis=0)

        def voigt_create(x,beta,alpha,pos,gamma):
            voigt_dist = beta*alpha *np.exp(-4*np.log(2)*(x-pos)**2 / (gamma**2)) + ((1-beta)*alpha*gamma**2 / ((x-pos)**2 + gamma**2))
            return voigt_dist
            


        wave_length = self.wl.astype('float')
        wl = self.wl.astype('float')





        #vx = sum([voigt_create(xx,np.random.random(1),np.random.randint(4,40),np.random.randint(30,950),np.random.randint(2,5)) for i in range(numer_of_peaks)])
        #vx = Data_array[0]
        vx_cut = np.copy(Data_array)

        peaks_id = find_peaks(vx_cut,height=self.threshold)[0]
        self.peaks_before = np.copy(peaks_id)
        duplicate_threshold = 1e-1
        duplicates = np.where(np.abs(np.diff(vx_cut))<=duplicate_threshold)[0]
        #peaks_id  = np.array(list(set(list(peaks_id)) -set(list(duplicates))))
        self.duplicates = np.copy(duplicates)
        self.peaks_after = np.copy(peaks_id)


        peaks_v = vx_cut[peaks_id]
        peaks_loc = wl[peaks_id]
        peaks_n = len(peaks_id)
        length = peaks_n *4
        
        input_=  vx_cut.reshape(1,-1)
        Labels = np.array((peaks_loc,peaks_v))#np.swapaxes(np.array((peaks_loc,peaks_v)),0,1)

        Labels=Labels[None,:]

        Target_shape = peaks_n
        self.target = Target_shape
        N = Labels.shape[0]


            
        input_shape = input_.shape[1]


        split=0
        split= 1 - split


        train_data = input_[:int(N*split)].astype(float)
        test_data = input_[int(N*split):].astype(float)
        train_labels = Labels[:int(N*split),:].astype(float)
        test_labels = Labels[int(N*split):,:].astype(float)


        train_data = input_.astype(float)
        train_labels = Labels.astype(float)

        train_labels_tensor =torch.from_numpy(train_labels).type(torch.FloatTensor) #torch.from_numpy(np.linspace(0,100,train_data.shape[0]))
        test_labels_tensor = torch.from_numpy(test_labels).type(torch.FloatTensor) #torch.from_numpy(np.linspace(0,100,test_data.shape[0]))



        train_set = torch.utils.data.TensorDataset(torch.from_numpy(train_data),train_labels_tensor) #
        test_set = torch.utils.data.TensorDataset(torch.from_numpy(test_data),test_labels_tensor)


        def setup_data_loaders(batch_size=128, use_cuda=False):
            root = './data'
            download = True
            trans = transforms.ToTensor()

            train_loader = torch.utils.data.DataLoader(dataset=train_set,
                batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(dataset=test_set,
                batch_size=batch_size, shuffle=False)
            return train_loader, test_loader

        class Net(nn.Module):
            def __init__(self):
                super().__init__()


                self.cvd1 = nn.Conv1d(1, 50, 10, stride=1)
                self.cvd2 = nn.Conv1d(50, 100, 5, stride=2)
                self.cvd3 = nn.Conv1d(100, 70, 5, stride=2)
                self.cvd4 = nn.Conv1d(70, 50, 5, stride=2)
                self.cvd5 = nn.Conv1d(50, 40, 5, stride=2)
                self.cvd6 = nn.Conv1d(40, 40, 5, stride=2)
                self.mp = nn.MaxPool1d(5, stride=2)
                before_out_num=1000

                self.BEFORE_OUT = nn.Linear(1000, before_out_num)
                #min(batch_s,N)*
                self.fc_out1 = nn.Linear(before_out_num, Target_shape)
                self.fc_out2 = nn.Linear(before_out_num, Target_shape)
                self.fc_out3 = nn.Linear(before_out_num, Target_shape)
                #self.fc16 = nn.Linear(hidden_dim, input_shape)


                self.main_activation = nn.Tanh()
                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
                self.sigmoid = nn.Sigmoid()
                self.Dout = nn.Dropout(0.25)
                self.threshold = 0.5

            def forward(self, x_in,targets,x):
                
                #print(x_in.get_device())
                hidden = self.cvd1(x_in)
                hidden = self.main_activation(self.mp(hidden))

                hidden = self.cvd2(hidden)
                hidden = self.main_activation(self.mp(hidden))
                hidden = self.cvd3(hidden)
                #hidden= self.Dout(hidden)
                hidden = self.main_activation(self.mp(hidden))
                hidden = self.cvd4(hidden)
                #hidden= self.Dout(hidden)
                hidden = self.main_activation(self.mp(hidden))
                #hidden = hidden.flatten().reshape(1,-1)
                hidden = self.cvd5(hidden)
                #hidden= self.Dout(hidden)
                hidden = self.main_activation(self.mp(hidden))
                hidden = self.cvd6(hidden)
                #hidden= self.Dout(hidden)
                hidden = self.main_activation(self.mp(hidden))
                hidden=  hidden.view(hidden.size(0), -1) 
                #print('shape')
                #print(hidden.shape)
                #hidden = self.main_activation(self.BEFORE_OUT(hidden))
                #self.output1 = torch.ones_like(self.sigmoid(self.fc_out1(hidden)))*0.5
                self.output1 = self.sigmoid(self.fc_out1(hidden))
                self.output2 = self.sigmoid(self.fc_out2(hidden)) #*0.05
                self.output = torch.cat([self.output1[:,:,None],self.output2[:,:,None]],dim=2)
                self.output = torch.swapaxes(self.output, 1, 2)

                return self.output




        class my_loss(nn.Module):
            def __init__(self, Lambda=10,p1=10,p2=1):
                super(my_loss, self).__init__()
                pass
            def forward(self,x, outputs1, outputs2, targets):        
                def part_it(x,beta_o,gamma_o,pos_o,alpha_o):
                    #print(len(beta_o))
                    voigt_dist_sum = torch.zeros(len(x)).cuda().requires_grad_(True)

                    if len(beta_o) > 1000:
                        frac = np.linspace(0,1,1000)
                        for i in range(len(frac)):
                            if i<3:
                                beta_it_o = beta_o[int(len(beta_o)*frac[i]):int(len(beta_o)*frac[i+1])]#.cpu()#.numpy()
                                gamma_it_o = gamma_o[int(len(beta_o)*frac[i]):int(len(beta_o)*frac[i+1])]#.cpu()#.numpy()
                                pos_it_o = pos_o[int(len(beta_o)*frac[i]):int(len(beta_o)*frac[i+1])]#.cpu()#.numpy()
                                alpha_it_o = alpha_o[int(len(beta_o)*frac[i]):int(len(beta_o)*frac[i+1])]#.cpu()#.numpy()
                                #print(len(pos_it_o))
                                voigt_dist = beta_it_o[:,None]*alpha_it_o[:,None]*torch.exp(-4*torch.log(torch.tensor(2))*(x-pos_it_o[:,None])**2 / (gamma_it_o[:,None]**2)) + ((1-beta_it_o[:,None])*alpha_it_o[:,None]*gamma_it_o[:,None]**2 / ((x-pos_it_o[:,None])**2 + gamma_it_o[:,None]**2))
                                voigt_dist_sum = voigt_dist_sum + voigt_dist.sum(axis=0)
                                
                    else:
                        voigt_dist = beta_o[:,None]*alpha_o[:,None]*torch.exp(-4*torch.log(torch.tensor(2))*(x-pos_o[:,None])**2 / (gamma_o[:,None]**2)) + ((1-beta_o[:,None])*alpha_o[:,None]*gamma_o[:,None]**2 / ((x-pos_o[:,None])**2 + gamma_o[:,None]**2))
                        voigt_dist_sum = voigt_dist_sum + voigt_dist.sum(axis=0)
                    
                    return voigt_dist_sum.requires_grad_(True)
                
                
                pos =  outputs1[:,0,:]#.flatten()
                alpha = outputs1[:,1,:]#.flatten()           
                
                
                beta = outputs2[:,0,:]#.flatten()
                gamma= outputs2[:,1,:]#.flatten()

                self.fitted_voigt = torch.zeros(len(pos),len(x)).cuda()

                for i in range(len(pos)):
                    beta_it = beta[i]#.cpu()#.numpy()
                    gamma_it = gamma[i]#.cpu()#.numpy()
                    pos_it = pos[i]#.cpu()#.numpy()
                    alpha_it = alpha[i]#.cpu()#.numpy()

                    #self.voigt_dist = beta_it[:,None]*alpha_it[:,None]*torch.exp(-4*torch.log(torch.tensor(2))*(x-pos_it[:,None])**2 / (gamma_it[:,None]**2)) + ((1-beta_it[:,None])*alpha_it[:,None]*gamma_it[:,None]**2 / ((x-pos_it[:,None])**2 + gamma_it[:,None]**2))
                    #fitted_voigt[i] = self.voigt_dist.sum(axis=0).clone()#.detach().requires_grad_(True)#torch.tensor(self.voigt_dist.sum(axis=0))
                    self.fitted_voigt[i] = part_it(x,beta_it,gamma_it,pos_it,alpha_it).clone()#self.voigt_dist.sum(axis=0).clone()
                    
                
                #part_it(x,beta_it,gamma_it,pos_it,alpha_it)
                
                
                loss =torch.mean((self.fitted_voigt - targets)**2)
                return loss



        net = Net()
        net.to('cuda')

        best_loss=1e6
        epoch_list = [15,self.max_epoch,self.max_epoch]
        learning_rate = [1e-3,1e-4,1e-5]
        for cnt,lr in enumerate(learning_rate):
            USE_CUDA = True
            Lambda = 0.1
            p1 = 1000
            p2 = 0.01
            epochs = epoch_list[cnt]
            batch_s = 1


            wl_c = torch.from_numpy(wl).type(torch.FloatTensor).cuda()
            #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
            optimizer = optim.Adam(net.parameters(), lr=lr)

            # create a loss function

            criterion = my_loss()
            #criterion = nn.MSELoss()
            log_interval = 10
            TEST_FREQUENCY = 5
            train_loader, test_loader = setup_data_loaders(batch_size=batch_s, use_cuda=USE_CUDA)

            for epoch in range(epochs):
                for i, (inputs, peaks) in enumerate(train_loader):
                    inputs = inputs.cuda()

                    inputs_channels,peaks = inputs.type(torch.FloatTensor)[:,None,:].cuda(), peaks.cuda()

                    coefs = net(inputs_channels,peaks,wl)

                    loss = criterion(wl_c,peaks,coefs, inputs)

                    optimizer.zero_grad()
                    loss.backward()
                    # update model weights
                    optimizer.step()
                    if loss<best_loss:
                    #print('saving')
                        torch.save(net.state_dict(), 'best-model-parameters.pt') 
                        best_loss = loss
                    if i % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, i * len(inputs), len(train_loader.dataset),
                                    100. * i / len(train_loader), loss.data))
        

            net.load_state_dict(torch.load('best-model-parameters.pt'))
            parameters1 = net(inputs_channels,peaks,wl)
            parameters1 = parameters1.detach().cpu().numpy()[0,:,:]
            parameters2 = Labels[0,:,:]
            parameters = np.vstack((parameters1,parameters2))
        return parameters

    def Reconstruct_voigt(self,wl,parameters):
        def voigt_vec(x,beta,alpha,pos,gamma):
            print(beta.shape)
            voigt_dist = beta[:,None]*alpha[:,None] *np.exp(-4*np.log(2)*(x-pos[:,None])**2 / (gamma[:,None]**2)) + ((1-beta[:,None])*alpha[:,None]*gamma[:,None]**2 / ((x-pos[:,None])**2 + gamma[:,None]**2))
            return voigt_dist.sum(axis=0)

        spectrum = voigt_vec(wl,parameters[0],parameters[3],parameters[2],parameters[1])
        return spectrum