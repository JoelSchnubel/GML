import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


torch.autograd.set_detect_anomaly(True)


class LSTM_QNET(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers ,output_size):
        super(LSTM_QNET,self).__init__()
        
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.output_layer =  nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x, _ = self.lstm(x)
        
        x = self.fc(x[-1, :])
        x = self.relu(x)
        x = self.output_layer(x)
        #return F.softmax(x,dim=0)
        return x
        
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


    




class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    #TODO for long term memory
    def train_step(self, state, action, reward, next_state,done,long):
        pred = []
        nexts = []
        if long:
            
            for s in state:  
                pred.append(self.model(torch.tensor(s,dtype=torch.float)))
            
            for s in next_state:
                nexts.append(self.model(torch.tensor(s,dtype=torch.float)))
                    
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
            
            
            
            
            
            target = pred.copy()
            for idx in range(len(pred)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx].item() + self.gamma * torch.max(nexts[idx]).item()
                

                target[idx][torch.argmax(action[idx]).item()] = Q_new
            
            
            
                # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
                # pred.clone()
                # preds[argmax(action)] = Q_new
            self.optimizer.zero_grad()
            loss = self.criterion(torch.stack(target),torch.stack(pred))
            loss.backward()

            self.optimizer.step()
        
            
                
        else:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
                
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
                    # (n, x)
            
                    #print("Lengths of sublists in state:", [len(sublist) for sublist in state])
                    #print(state)
                
            if len(state.shape) == 2:
                    # (1, x)
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done, )

                # 1: predicted Q values with current state
            pred = self.model(state)
            
                
            target = pred.clone()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx].item() + self.gamma * torch.max(self.model(next_state[idx])).item()
                

                target[idx][torch.argmax(action[idx]).item()] = Q_new
          
            # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
            # pred.clone()
            # preds[argmax(action)] = Q_new
            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()

            self.optimizer.step()
        