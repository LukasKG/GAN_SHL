# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ is None or __package__ == '':
    from metrics import calc_accuracy, calc_f1score
    import network
    import preprocessing as pp
else:
    from .metrics import calc_accuracy, calc_f1score
    from . import network
    from . import preprocessing as pp
   
def train_Base(P, DL_L, DL_V, name=None):   
    name = name if name is not None else P.get('name')

    R_Loss = network.CrossEntropyLoss_OneHot()
        
    if P.get('CUDA') and torch.cuda.is_available():
        R_Loss.cuda()

    mat_accuracy = np.zeros((1, int(P.get('epochs')/P.get('save_step'))+1))
    mat_f1_score = np.zeros((1, int(P.get('epochs')/P.get('save_step'))+1))
    R = network.load_R(P)  
    optimizer_R = network.get_optimiser(P,'C',R.parameters())
    
    for epoch in range(P.get('epochs_GD'),P.get('epochs')):
        R.train()
        for i, (X1, Y1) in enumerate(DL_L, 1):
            optimizer_R.zero_grad()
            PR = R(X1)
            loss = R_Loss(PR, Y1)
            loss.backward()
            optimizer_R.step()
            
        if (epoch+1)%P.get('save_step') == 0:
            idx = int(epoch/P.get('save_step'))+1
            R.eval()
            with torch.no_grad():
                mat1, mat2 = [], []
                for XV, YV in DL_V:
                    mat1.append(calc_accuracy(R(XV), YV))
                    mat2.append(calc_f1score(R(XV), YV))
                mat_accuracy[0,idx] = np.mean(mat1)
                mat_f1_score[0,idx] = np.mean(mat2)
    return R, mat_accuracy, mat_f1_score
 
def train_GD(P, DL_L, DL_V, mat_accuracy=None, mat_f1_score=None, name=None):
    name = name if name is not None else P.get('name')
        
    D_Loss = torch.nn.BCELoss()

    if P.get('CUDA') and torch.cuda.is_available():
        D_Loss.cuda()
        floatTensor = torch.cuda.FloatTensor
    else:
        floatTensor = torch.FloatTensor

    if mat_accuracy is None:
        mat_accuracy = np.zeros((2, int(P.get('epochs_GD')/P.get('save_step'))+1))
    if mat_f1_score is None:
        mat_f1_score = np.zeros((2, int(P.get('epochs_GD')/P.get('save_step'))+1))

    G = network.load_G(P)
    D = network.load_D(P)

    optimizer_G = network.get_optimiser(P,'G',G.parameters())
    optimizer_D = network.get_optimiser(P,'D',D.parameters())
    
    for epoch in range(P.get('epochs_GD')):
        
        if P.get('print_epoch'):
            loss_G = []
            loss_D = []
        
        G.train();D.train();
        for i, (X1, Y1) in enumerate(DL_L, 1):
            
            # -------------------
            #  Train the discriminator to label real samples
            # -------------------
            W1 = torch.cat((X1,Y1),dim=1)
            R1 = floatTensor(W1.shape[0], 1).fill_(1.0)
            
            optimizer_D.zero_grad()
            A1 = D(W1)
            loss = D_Loss(A1, R1)
            loss.backward()
            optimizer_D.step()
            if P.get('print_epoch'):
                loss_D.append(loss)
            
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
            loss = D_Loss(A3, R3)
            loss.backward()
            optimizer_G.step()
            if P.get('print_epoch'):
                loss_G.append(loss)
            
            # -------------------
            #  Train the discriminator to label synthetic samples
            # -------------------
            optimizer_D.zero_grad()
            A3 = D(W3.detach())
            F3 = floatTensor(W3.shape[0], 1).fill_(0.0)
            loss = D_Loss(A3, F3)
            loss.backward()
            optimizer_D.step()
            if P.get('print_epoch'):
                loss_D.append(loss)
            
        
        # -------------------
        #  Post Epoch
        # -------------------
        
        if P.get('print_epoch'):
            logString = f"[Epoch {epoch+1}/{P.get('epochs')}] [G loss: {np.mean([loss.item() for loss in loss_G])}] [D loss: {np.mean([loss.item() for loss in loss_D])}]"
            P.log(logString,save=False)
        
        if (epoch+1)%P.get('save_step') == 0:
            idx = int(epoch/P.get('save_step'))+1
            
            D_acc = np.zeros((2,len(DL_V)))

            D_f1 = np.zeros((2,len(DL_V))) 
            G_f1 = np.zeros((len(DL_V)))

            G.eval();D.eval()
            
            with torch.no_grad():
                for i, (XV, YV) in enumerate(DL_V):
                    
                    # Generate Synthetic Data
                    Z = floatTensor(np.random.normal(0, 1, (YV.shape[0], P.get('noise_shape'))))
                    IV = torch.cat((Z,YV),dim=1)
                    XG = G(IV)
                    
                    # Estimate Discriminator Accuracy
                    WV1 = torch.cat((XV,YV),dim=1)
                    WV3 = torch.cat((XG,YV),dim=1)
                    RV1 = floatTensor(WV1.shape[0],1).fill_(1.0)
                    FV3 = floatTensor(WV3.shape[0],1).fill_(0.0)
                    RV3 = floatTensor(WV3.shape[0],1).fill_(1.0)
                    
                    AV1 = D(WV1)
                    AV3 = D(WV3)
                    
                    D_acc[0,i] = calc_accuracy(AV1,RV1)
                    D_acc[1,i] = calc_accuracy(AV3,FV3)
                                         
                    D_f1[0,i] = calc_f1score(AV1,RV1)
                    D_f1[1,i] = calc_f1score(AV3,FV3)

                    G_f1[i] = calc_f1score(AV3,RV3)

                    
            D_acc = np.mean(D_acc,axis=1)    
            acc_D = np.mean(D_acc)
            mat_accuracy[1,idx] = acc_D
        
            acc_G = 1.0 - D_acc[1]
            mat_accuracy[0,idx] = acc_G

            mat_f1_score[0,idx] = np.mean(G_f1)
            mat_f1_score[1,idx] = np.mean(D_f1)
            
            logString = f"[{name}] [Epoch {epoch+1}/{P.get('epochs')}] [G F1: {mat_f1_score[0,idx]} acc: {acc_G}] [D F1: {mat_f1_score[1,idx]} acc: {acc_D} | vs Real: {D_acc[0]} | vs G: {D_acc[1]}]"
            P.log(logString,save=True) 
            
    return G, D, mat_accuracy, mat_f1_score
        
def train_GAN(P, DL_L, DL_U_iter, DL_V, name=None):
    name = name if name is not None else P.get('name')
    
    # -------------------
    #  Parameters
    # -------------------
    
    #P.log(str(P))
    plt.close('all')
    
    # -------------------
    #  CUDA
    # -------------------
    
    D_Loss = torch.nn.BCELoss()
    C_Loss = network.CrossEntropyLoss_OneHot()

    if P.get('CUDA') and torch.cuda.is_available():
        D_Loss.cuda()
        C_Loss.cuda()
        floatTensor = torch.cuda.FloatTensor
        #P.log("CUDA Training.")
        #network.clear_cache()
    else:
        floatTensor = torch.FloatTensor
        #P.log("CPU Training.")

    # -------------------
    #  Metrics
    # -------------------

    mat_accuracy = np.zeros((3, int((P.get('epochs'))/P.get('save_step'))+1))
    mat_f1_score = np.zeros((3, int((P.get('epochs'))/P.get('save_step'))+1))
        
    # -------------------
    #  Networks
    # -------------------
    
    G, D, mat_accuracy, mat_f1_score = train_GD(P, DL_L, DL_V, mat_accuracy, mat_f1_score, name)
    C = network.load_C(P)

    # -------------------
    #  Optimizers
    # -------------------
    
    optimizer_G = network.get_optimiser(P,'G',G.parameters())
    optimizer_D = network.get_optimiser(P,'D',D.parameters())
    optimizer_C = network.get_optimiser(P,'C',C.parameters())
    
    # -------------------
    #  Training
    # -------------------
    
    for epoch in range(P.get('epochs_GD'),P.get('epochs')):
        
        if P.get('print_epoch'):
            running_loss_G = 0.0
            running_loss_D = 0.0
            running_loss_C = 0.0
        
        G.train();D.train();C.train();
        
        """
              X1, P1      - Labelled Data,      predicted Labels (C)                             | Regular training of classifier
        W1 = (X1, Y1), A1 - Labelled Data,      actual Labels,        predicted Authenticity (D) | Real samples
        W2 = (X2, Y2), A2 - Unlabelled Data,    predicted Labels (C), predicted Authenticity (D) | Real data with fake labels
        W3 = (X3, Y3), A3 - Synthetic Data (G), actual Labels,        predicted Authenticity (D) | Fake data with real labels
        W4 = (X4, Y4), A4 - Unlabbeled Data,    predicted Labels (C), predicted Authenticity (D) | Fake positive to prevent overfitting
              XV, YV,  PV - Validation Data,    actual Labels,        predicted Labels (C)       | Validation samples
          R1, F2, F3,  R4 - Real/Fake Labels
        """
        for i, (X1, Y1) in enumerate(DL_L, 1):
            
            if P.get('print_epoch'):
                loss_G = []
                loss_D = []
                loss_C = []
            
            # -------------------
            #  Train the classifier on real samples
            # -------------------
            W1 = torch.cat((X1,Y1),dim=1)
            R1 = floatTensor(W1.shape[0], 1).fill_(1.0)
            
            if P.get('C_basic_train'):
                optimizer_C.zero_grad()
                P1 = C(X1)
                loss = C_Loss(P1, Y1)
                loss.backward()
                optimizer_C.step()
                if P.get('print_epoch'):
                    loss_C.append(loss)
                
            # -------------------
            #  Train the discriminator to label real samples
            # -------------------
            optimizer_D.zero_grad()
            A1 = D(W1)
            loss = D_Loss(A1, R1)
            loss.backward()
            optimizer_D.step()
            if P.get('print_epoch'):
                loss_D.append(loss)
            
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
            loss = D_Loss(A2, R2)
            loss.backward()
            optimizer_C.step()
            if P.get('print_epoch'):
                loss_C.append(loss)
            
            # -------------------
            #  Train the discriminator to label predicted samples
            # -------------------
            optimizer_D.zero_grad()
            A2 = D(W2.detach())
            F2 = floatTensor(W2.shape[0], 1).fill_(0.0)
            loss = D_Loss(A2, F2)
            loss.backward()
            optimizer_D.step()
            if P.get('print_epoch'):
                loss_D.append(loss)
            
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
            loss.backward()
            optimizer_D.step()
            if P.get('print_epoch'):
                loss_D.append(loss)
            
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
            loss = D_Loss(A3, R3)
            loss.backward()
            optimizer_G.step()
            if P.get('print_epoch'):
                loss_G.append(loss)
            
            # -------------------
            #  Train the discriminator to label synthetic samples
            # -------------------
            optimizer_D.zero_grad()
            A3 = D(W3.detach())
            F3 = floatTensor(W3.shape[0], 1).fill_(0.0)
            loss = D_Loss(A3, F3)
            loss.backward()
            optimizer_D.step()
            if P.get('print_epoch'):
                loss_D.append(loss)
            
            # -------------------
            #  Calculate overall loss
            # -------------------
            if P.get('print_epoch'):
                running_loss_G += np.mean([loss.item() for loss in loss_G])
                running_loss_D += np.mean([loss.item() for loss in loss_D])
                running_loss_C += np.mean([loss.item() for loss in loss_C])
        
        # -------------------
        #  Post Epoch
        # -------------------
        
        if P.get('print_epoch'):
            logString = f"[Epoch {epoch+1}/{P.get('epochs')}] [G loss: {running_loss_G/(i)}] [D loss: {running_loss_D/(i)}] [C loss: {running_loss_C/(i)}]"
            P.log(logString,save=False)
        
        if (epoch+1)%P.get('save_step') == 0:
            idx = int(epoch/P.get('save_step'))+1
            
            D_acc = np.zeros((3,len(DL_V)))
            C_acc = np.zeros(len(DL_V))

            D_f1 = np.zeros((3,len(DL_V))) 
            C_f1 = np.zeros((len(DL_V)))
            G_f1 = np.zeros((len(DL_V)))

            G.eval();D.eval();C.eval();
            
            with torch.no_grad():
                for i,data in enumerate(DL_V):
                    
                    XV, YV = data
                
                    # Predict labels
                    PV = C(XV)
                    
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
                    RV3 = floatTensor(WV3.shape[0],1).fill_(1.0)
                    
                    AV1 = D(WV1)
                    AV2 = D(WV2)
                    AV3 = D(WV3)
                    
                    D_acc[0,i] = calc_accuracy(AV1,RV1)
                    D_acc[1,i] = calc_accuracy(AV2,FV2)
                    D_acc[2,i] = calc_accuracy(AV3,FV3)
                                        
                    C_acc[i] = calc_accuracy(PV, YV)
                    
                    D_f1[0,i] = calc_f1score(AV1,RV1, average = 'micro')
                    D_f1[1,i] = calc_f1score(AV2,FV2, average = 'micro')
                    D_f1[2,i] = calc_f1score(AV3,FV3, average = 'micro')

                    C_f1[i] = calc_f1score(PV, YV, average = 'micro')
                    G_f1[i] = calc_f1score(AV3,RV3, average = 'micro')

                    
            D_acc = np.mean(D_acc,axis=1)    
            acc_D = .5*D_acc[0] + .25*D_acc[1] + .25*D_acc[2]
            mat_accuracy[1,idx] = acc_D
        
            acc_C_real = np.mean(C_acc)
            acc_C_vs_D = 1.0 - D_acc[1]
            acc_C = .5*acc_C_real + .5*acc_C_vs_D
            mat_accuracy[2,idx] = acc_C_real
        
            acc_G = 1.0 - D_acc[2]
            mat_accuracy[0,idx] = acc_G

            mat_f1_score[0,idx] = np.mean(G_f1)
            mat_f1_score[1,idx] = np.mean(D_f1)
            mat_f1_score[2,idx] = np.mean(C_f1)
            
            logString = f"[{name}] [Epoch {epoch+1}/{P.get('epochs')}] [G F1: {mat_f1_score[0,idx]} acc: {acc_G}] [D F1: {mat_f1_score[1,idx]} acc: {acc_D} | vs Real: {D_acc[0]} | vs G: {D_acc[2]} | vs C: {D_acc[1]}] [C F1: {mat_f1_score[2,idx]} acc: {acc_C} | vs Real: {acc_C_real} | vs D: {acc_C_vs_D}]"
            P.log(logString,save=True) 
                   
    # -------------------
    #  Post Training
    # -------------------
    
    if P.get('save_GAN'):
        network.save_GAN(name,G,D,C)
            
    return G, D, C, mat_accuracy, mat_f1_score
    
if __name__ == "__main__":
    import main
    main.main()