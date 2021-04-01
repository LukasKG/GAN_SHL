# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ is None or __package__ == '':
    import network
    from log import log
    import preprocessing as pp
else:
    from . import network
    from .log import log
    from . import preprocessing as pp
   
    
def get_accuracy(Y_pred,Y_true):
    if Y_true.size()[1] == 1:
        return (Y_pred.round() == Y_true).sum().item() / Y_true.size(0)
    else:
        return (Y_pred.max(dim=1)[1] == Y_true.max(dim=1)[1]).sum().item() / Y_true.size(0)
    
def train_GAN(P, DL_L, DL_U_iter, DL_V, name=None):
    
    # -------------------
    #  Parameters
    # -------------------
    
    #log(str(P),name=P.get('log_name'))
    if name is None:
        name = P.get('name')
    plt.close('all')
    
    # -------------------s
    #  CUDA
    # -------------------
    
    G_Loss = torch.nn.BCELoss() # might try logitsloss
    D_Loss = torch.nn.BCELoss()
    C_Loss = torch.nn.BCELoss()

    if P.get('CUDA') and torch.cuda.is_available():
        G_Loss.cuda()
        D_Loss.cuda()
        C_Loss.cuda()
        floatTensor = torch.cuda.FloatTensor
        #log("CUDA Training.",name=P.get('log_name'))
        #network.clear_cache()
    else:
        floatTensor = torch.FloatTensor
        #log("CPU Training.",name=P.get('log_name'))

    # -------------------
    #  Accuracy
    # -------------------

    mat_accuracy = np.zeros((4 if P.get('R_active') else 3, int(P.get('epochs')/P.get('save_step'))+1))
        
    # -------------------
    #  Networks
    # -------------------
    
    G, D, C = network.load_GAN(P,name)
    
    R = None
    if(P.get('R_active')):
        R = network.load_Ref(P,name)
        
    # -------------------
    #  Optimizers
    # -------------------
    
    optimizer_G = network.get_optimiser(P,'G',G.parameters())
    optimizer_D = network.get_optimiser(P,'D',D.parameters())
    optimizer_C = network.get_optimiser(P,'C',C.parameters())
    
    if(P.get('R_active')):
        optimizer_R = network.get_optimiser(P,'C',R.parameters())
    
    # -------------------
    #  Training
    # -------------------

    for epoch in range(P.get('epochs')):
        
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_C = 0.0
        
        """
              X1, P1      - Labelled Data,      predicted Labels (C)                             | Regular training of classifier
        W1 = (X1, Y1), A1 - Labelled Data,      actual Labels,        predicted Authenticity (D) | Real samples
        W2 = (X2, Y2), A2 - Unlabelled Data,    predicted Labels (C), predicted Authenticity (D) | Real data with fake labels
        W3 = (X3, Y3), A3 - Synthetic Data (G), actual Labels,        predicted Authenticity (D) | Fake data with real labels
        W4 = (X4, Y4), A4 - Unlabbeled Data,    predicted Labels (C), predicted Authenticity (D) | Fake positive to prevent overfitting
              XV, YV,  PV - Validation Data,    actual Labels,        predicted Labels (C)       | Validation samples
          R1, F2, F3,  R4 - Real/Fake Labels
        """
        for i, data in enumerate(DL_L, 1):
            
            loss_G = []
            loss_D = []
            loss_C = []
            
            # -------------------
            #  Train the classifier on real samples
            # -------------------
            X1, Y1 = data
            W1 = torch.cat((X1,Y1),dim=1)
            R1 = floatTensor(W1.shape[0], 1).fill_(1.0)
            
            if P.get('C_basic_train'):
                optimizer_C.zero_grad()
                P1 = C(X1)
                loss = C_Loss(P1, Y1)
                loss_C.append(loss)
                loss.backward()
                optimizer_C.step()
            
            if P.get('R_active'):
                optimizer_R.zero_grad()
                PR = R(X1)
                loss = C_Loss(PR, Y1)
                loss.backward()
                optimizer_R.step()
                
            # -------------------
            #  Train the discriminator to label real samples
            # -------------------
            optimizer_D.zero_grad()
            A1 = D(W1)
            loss = D_Loss(A1, R1)
            loss_D.append(loss)
            loss.backward()
            optimizer_D.step()
            
            # -------------------
            #  Classify unlabelled data
            # -------------------
            optimizer_C.zero_grad()
            X2 = DL_U_iter.get_next()[0]
            Y2 = C(X2)
            W2 = torch.cat((X2,Y2),dim=1)

            # -------------------
            #  Train the classifier to label unlabelled samples
            # -------------------
            A2 = D(W2)
            R2 = floatTensor(W2.shape[0], 1).fill_(1.0)
            loss = C_Loss(A2, R2)
            loss_C.append(loss)
            loss.backward()
            optimizer_C.step()
            
            # -------------------
            #  Train the discriminator to label predicted samples
            # -------------------
            optimizer_D.zero_grad()
            A2 = D(W2.detach())
            F2 = floatTensor(W2.shape[0], 1).fill_(0.0)
            loss = D_Loss(A2, F2)
            loss_D.append(loss)
            loss.backward()
            optimizer_D.step()
            
            # -------------------
            #  Train the discriminator to label fake positive samples
            # -------------------
            X4 = DL_U_iter.get_next()[0]
            Y4 = C(X4)
            W4 = torch.cat((X4,Y4),dim=1)
            
            optimizer_D.zero_grad()
            A4 = D(W4.detach())
            R4 = floatTensor(W4.shape[0], 1).fill_(1.0)
            loss = D_Loss(A4, R4)
            loss_D.append(loss)
            loss.backward()
            optimizer_D.step()
            
            # -------------------
            #  Create Synthetic Data
            # -------------------     
            optimizer_G.zero_grad()
            if P.get('G_label_sample'):
                # Selected Labels from a uniform distribution of available labels
                Y3 = floatTensor(pp.get_one_hot_labels(P,num=Y1.shape[0]*P.get('G_label_factor')))
            else:
                # Select labels from current training batch
                Y3 = torch.cat(([Y1 for _ in range(P.get('G_label_factor'))]),dim=0)
            
            Z = floatTensor(np.random.normal(0, 1, (Y3.shape[0], P.get('noise_shape'))))
            I3 = torch.cat((Z,Y3),dim=1)
            X3 = G(I3)
            W3 = torch.cat((X3,Y3),dim=1)
            
            # -------------------
            #  Train the generator to fool the discriminator
            # -------------------
            A3 = D(W3)
            R3 = floatTensor(W3.shape[0], 1).fill_(1.0)
            loss = G_Loss(A3, R3)
            loss_G.append(loss)
            loss.backward()
            optimizer_G.step()
            
            # -------------------
            #  Train the discriminator to label synthetic samples
            # -------------------
            optimizer_D.zero_grad()
            A3 = D(W3.detach())
            F3 = floatTensor(W3.shape[0], 1).fill_(0.0)
            loss = D_Loss(A3, F3)
            loss_D.append(loss)
            loss.backward()
            optimizer_D.step()
            
            # -------------------
            #  Calculate overall loss
            # -------------------
            running_loss_G += np.mean([loss.item() for loss in loss_G])
            running_loss_D += np.mean([loss.item() for loss in loss_D])
            running_loss_C += np.mean([loss.item() for loss in loss_C])
        
        # -------------------
        #  Post Epoch
        # -------------------
        
        if P.get('print_epoch'):
            logString = "[Epoch %d/%d] [G loss: %f] [D loss: %f] [C loss: %f]"%(epoch+1, P.get('epochs'), running_loss_G/(i), running_loss_D/(i), running_loss_C/(i))
            log(logString,save=False,name=P.get('log_name'))
        
        if (epoch+1)%P.get('save_step') == 0:
            idx = int(epoch/P.get('save_step'))+1
            
            acc_D_real = []
            acc_D_vs_C = []
            acc_D_vs_G = []
            acc_C_real = []
            
            for data in DL_V:
                
                XV, YV = data
            
                # Predict labels
                PV = C(XV)

                if P.get('R_active'):
                    PR = R(XV)
                    mat_accuracy[3,idx] = get_accuracy(PR, YV)
                
                # Generate Synthetic Data
                Z = floatTensor(np.random.normal(0, 1, (YV.shape[0], P.get('noise_shape'))))
                IV = torch.cat((Z,YV),dim=1)
                XG = G(IV)
                
                # Estimate Discriminator Accuracy
                WV1 = torch.cat((XV,YV),dim=1)
                WV2 = torch.cat((XV,PV),dim=1)
                WV3 = torch.cat((XG,YV),dim=1)
                RV1 = floatTensor(WV1.shape[0],1).fill_(1.0)
                FV2 = floatTensor(WV2.shape[0],1).fill_(0.0)
                FV3 = floatTensor(WV3.shape[0],1).fill_(0.0)
                
                AV1 = D(WV1)
                AV2 = D(WV2)
                AV3 = D(WV3)
                
                acc_D_real.append(get_accuracy(AV1,RV1))
                acc_D_vs_C.append(get_accuracy(AV2,FV2))
                acc_D_vs_G.append(get_accuracy(AV3,FV3))
            
                acc_C_real.append(get_accuracy(PV, YV))
 
            acc_D_real = np.mean(acc_D_real)
            acc_D_vs_C = np.mean(acc_D_vs_C)
            acc_D_vs_G = np.mean(acc_D_vs_G)
            acc_D = .5*acc_D_real + .25*acc_D_vs_G + .25*acc_D_vs_C
            mat_accuracy[1,idx] = acc_D
        
            acc_C_real = np.mean(acc_C_real)
            acc_C_vs_D = 1.0 - acc_D_vs_C
            acc_C = .5*acc_C_real + .5*acc_C_vs_D
            mat_accuracy[2,idx] = acc_C_real
        
            acc_G = 1.0 - acc_D_vs_G
            mat_accuracy[0,idx] = acc_G

            logString = "[%s] [Epoch %d/%d] [G acc: %f] [D acc: %f | vs Real: %f | vs G: %f | vs C: %f] [C acc: %f | vs Real: %f | vs D: %f]"%(name, epoch+1, P.get('epochs'), acc_G, acc_D, acc_D_real, acc_D_vs_G, acc_D_vs_C, acc_C, acc_C_real, acc_C_vs_D)
            log(logString,save=True,name=P.get('log_name')) 
                   
    # -------------------
    #  Post Training
    # -------------------
    
    if P.get('save_GAN'):
        network.save_GAN(name,G,D,C,R)
            
    return mat_accuracy, G, D, C, R
    
if __name__ == "__main__":
    import main
    main.main()