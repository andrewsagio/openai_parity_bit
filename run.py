import torch
import torch.nn as nn
import torch.optim as optim

SEQ_LEN = 50   # sequence length of training data
VAL_SEQ_LEN = 50  # sequence length of validation data
BATCH_SIZE = 8   # batch size of training
HIDDEN_SIZE = 12 # Number of hidden units in LSTM
NUM_LAYERS = 1  # Number of layers in LSTM
VARIABLE_LEN = False   # Use training data with variable lengths

class LSTM(nn.Module):
    ''' LSTM followed by a linear layer with output 1'''
    def __init__(self, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=self.hidden_size, 
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.hidden_size ,1)
    
    def forward(self, data):
        x, (h_n, c_n) = self.lstm(data)
        x = self.linear(x)
        return x
        
torch.manual_seed(0)
data = torch.randint(2, (100000, SEQ_LEN, 1)).float()
data_val = torch.randint(2, (1000, VAL_SEQ_LEN, 1)).float()

if VARIABLE_LEN:
    for i in range(len(data)):
        l = torch.randint(SEQ_LEN-1, (1,1))
        data[i, l:,0] = 0.        

targets = (data.cumsum(1)%2).long().squeeze()
targets_val = (data_val.cumsum(1)%2).long().squeeze()

loss_function = nn.BCEWithLogitsLoss()
m = LSTM(HIDDEN_SIZE, NUM_LAYERS)
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
m.to(dev)
data=data.to(dev)
targets=targets.to(dev)
data_val=data_val.to(dev)
targets_val=targets_val.to(dev)
optimizer = optim.SGD(m.parameters(), lr=1)
#optimizer = optim.Adam(m.parameters(), lr=.1)

step = 0
total_loss = 0.
for epoch in range(50):    
    for b in range(0,data.shape[0], BATCH_SIZE):
        
        batch_data = data[b: min(b+BATCH_SIZE,data.shape[0] )]
        batch_target = targets[b: min(b+BATCH_SIZE,data.shape[0])]
        step += len(batch_data)
        optimizer.zero_grad() 
        out=m(batch_data.view(-1,batch_data.shape[1],1))
        loss = loss_function(out.squeeze(), batch_target.float())
        loss.backward()
        optimizer.step()
        total_loss += loss
        
        if step >= 10000:
            with torch.no_grad():

                pred = torch.round(torch.sigmoid(out)).squeeze()
                acc_test = sum(pred == batch_target).float().mean()/len(pred)
                
                out = m(data_val)
                pred = torch.round(torch.sigmoid(out)).squeeze()
                acc_val = sum(pred == targets_val).float().mean()/len(pred)
                
                total_loss /= float(step)
                print('epoch %d, loss %1.8f, acc %1.2f, acc_val %1.2f'%(epoch, total_loss, acc_test, acc_val))
                total_loss = 0
                step = 0
                if acc_val == 1.:
                    # early stop
                    print('Training completed with %d samples'%(epoch*data.shape[0]+ b+len(batch_data)))
                    break
    
    if acc_val == 1.:
        break
        
    
    
        