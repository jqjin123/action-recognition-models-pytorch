import torch
import torch.nn as nn
import torchvision

class conv_pooling(nn.Module):
    # Input size: 120x224x224
    # The CNN structure is first trained from single frame, then the FC layers are fine-tuned from scratch.
    def __init__(self, num_class):
        super(conv_pooling, self).__init__()

        self.conv=nn.Sequential(* list(torchvision.models.resnet101().children())[:-2])
        #self.time_pooling=nn.MaxPool3d(kernel_size=(120,1,1))
        self.time_pooling = nn.MaxPool3d(kernel_size=(30, 1, 1))
        self.average_pool=nn.AvgPool3d(kernel_size=(1,7,7))
        self.linear1=nn.Linear(2048,2048)
        self.linear2=nn.Linear(2048, num_class)
    def forward(self, x):
        t_len=x.size(2)  # [batchSize, channel, tLen, width, height]
        conv_out_list=[]
        for i in range(t_len):
            conv_out_list.append(self.conv(torch.squeeze(x[:,:,i,:,:])))  # 每一个元素的维度都是 [batchSize, channel, width, height]
        stack_out=torch.stack(conv_out_list,2)  # [batchSize, channel, tLen, width, height]
        conv_out=self.time_pooling(stack_out)  # [batchSize, channel, 1, width, height]
        conv_out = self.average_pool(conv_out)  # # [batchSize, 2048, 1, 1, 1]
        conv_out=self.linear1(conv_out.view(conv_out.size(0),-1))
        conv_out=self.linear2(conv_out)
        return conv_out

class cnn_lstm(nn.Module):
    # Input size: 30x224x224
    # The CNN structure is first trained from single frame, then the lstm is fine-tuned from scratch.
    def __init__(self, num_class):
        super(cnn_lstm, self).__init__()

        self.conv = nn.Sequential(*list(torchvision.models.resnet101().children())[:-1])
        self.lstm = nn.LSTM(2048,512,5,batch_first=True)  # [inputSize, hiddenSize, layerNum]
        self.fc=nn.Linear(512,num_class)

    def forward(self, x):
        t_len = x.size(2)
        conv_out_list = []
        for i in range(t_len):
            conv_out_list.append(self.conv(torch.squeeze(x[:, :, i, :, :])))  # 每一个元素的维度都是[batchSize, channel, width, height], 此处为[2, 2048, 1, 1]
        conv_out=torch.stack(conv_out_list,1)  # [batchSize, tLen, channel, width, height] 此处为[2, 30, 2048, 1, 1]
        view_out=conv_out.view(conv_out.size(0), conv_out.size(1), -1)  # [batchSize, tLen, channel*width*height] 此处为[2, 30, 2048]
        conv_out,hidden=self.lstm(view_out)  # conv_out的维度为[batchSize, tLen, hiddenSize], 此处为[2, 30, 512]。hidden的维度为([layerNum, batchSize, hiddenSize], [layerNum, batchSize, hiddenSize]), 此处为([5, 2, 512], [5, 2, 512])
        lstm_out=[]
        for j in range (conv_out.size(1)):
            lstm_out.append(self.fc(torch.squeeze(conv_out[:,j,:])))  # lstm_out每个元素的维度为[batchSize, num_class]
        return torch.stack(lstm_out,1),hidden

if __name__ == '__main__':
    num_class = 10  # 假设的类别数量，需要根据实际情况调整
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查CUDA是否可用
    # model = conv_pooling(num_class).to(device)
    model = cnn_lstm(num_class).to(device)
    batch_size = 2
    t_len = 30
    width = 224
    height = 224
    input_shape = (batch_size, 3, t_len, width, height)
    input_tensor = torch.randn(input_shape).to(device)
    # output = model(input_tensor)
    output, hidden = model(input_tensor)
    print(output.shape, hidden[0].shape, hidden[1].shape)