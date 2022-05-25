# Copied & modified from https://www.dacon.io/competitions/official/235757/codeshare/3244?page=1&dtype=recent


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding_table(n_seq, hidn):
    
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / hidn)

    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(hidn)]
    
    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    # 홀수일때는 cosine 값을, 짝수일 때는 sine값을 할당하도록 코드 구성, shape(2 , 512)
    return sinusoid_table


def get_attn_decoder_mask(seq):
    batch, window_size, d_hidn = seq.size()
    subsequent_mask = torch.ones((batch, window_size, window_size), device=seq.device)
    subsequent_mask = subsequent_mask.triu(diagonal=1)
    return subsequent_mask


class scaled_dot_product_attention(nn.Module): #<- multihead별로 self_attention을 위한 코드(논문에서 scaled-dot attn이라고 명명)
    def __init__(self, args):
        super().__init__()
        self.args = args #argument를 그대로 가져다 씀(transformer는 parameter나 dimension을 유지하는 특성이 있어 효과적인 방법으로 보임)
        self.dropout = nn.Dropout(self.args.dropout)
        # attention score의 값이 너무 커지지 않도록(gradient vanishing의 발생가능성 때문, dk root를 multiply 해주는 과정임)
        self.scale = 1 / (self.args.d_head ** 0.5) # root of dk로 나눠주는 코드

    def forward(self, q, k, v, attn_mask=False):

        # transpose된 k와 matmul해주는 코드
        scores = torch.matmul(q, k.transpose(-1, -2))

        ## attention score를 구하는 과정. 이를 v에다 곱해줘서 result를 추출하게 됨.

        scores = scores.mul_(self.scale)  
        # scaling한 값이 바로 저장되도록 하는 메서드 
        # In-Place operation을 사용하면 계산 결과를 리턴하고, 그 값을 해당 변수에 저장한다.
        # In-Place operation을 하는 방법은 메소드 뒤에 _(언더바)를 붙이는 것이다.


        if attn_mask is not False:
            scores.masked_fill_(attn_mask, -1e9)
            attn_prob = nn.Softmax(dim=-1)(scores)
            attn_prob = self.dropout(attn_prob)
            context = torch.matmul(attn_prob, v)
        else:
            ## attention scores에다가 softmax씌운 값이랑, v vector와 matmul해서 attn_prob 도출. output dimension은 q,k,v의 input dimension과 값이 동일할 것. 
            attn_prob = nn.Softmax(dim=-1)(scores)
            attn_prob = self.dropout(attn_prob)
            context = torch.matmul(attn_prob, v)
    
        ## attention probability는 딴데 어디다 써먹어서 return받는 것일까? ㅠㅠㅠ
        return context, attn_prob

class multihead_attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.W_Q = nn.Linear(self.args.d_hidn, self.args.n_head * self.args.d_head) # <-nnlayer가 됨 fC layer를 만들어주는 코드
        self.W_K = nn.Linear(self.args.d_hidn, self.args.n_head * self.args.d_head)
        self.W_V = nn.Linear(self.args.d_hidn, self.args.n_head * self.args.d_head)

        self.scaled_dot_attn = scaled_dot_product_attention(self.args)
        self.linear = nn.Linear(self.args.n_head * self.args.d_head, self.args.d_hidn) #<-encoder layer의 output dimension을 본래의 dimension으로 축소시키기 위한 FC
        self.dropout = nn.Dropout(self.args.dropout)
    
    def forward(self, q, k, v, attn_mask=False): # <- query, key, value 값을 생성하는 구문. 값이 어떻게 생성되는지 확인할 것(기본적으로 batch_size를 앞에 달고 있네.)
        batch_size = q.size(0)

        # -1을 지정하는 경우는 dimension이 원래 차원에 맞게 자동지정됨. 단 -1을 두개 이상 포함하거나, 변환이 불가능한 경우에는
        # 두 함수 모두 run time error가 발생하게 됨. 

        '''
        df.reshape과 df.view의 차이를 정리할 것
        '''

        ## batch_size가 128, n_head: number of head(로 추정), d_head: head당 dimension: query를 표현하는 차원 수? => d model=n_head*d_head일 것.
        ## batch_size=128로, n_head:8, d_head=32이므로, 2번째 차원의 값은 2가 될 것 (shape 128,2,8,32)

        q_s = self.W_Q(q).view(batch_size, -1, self.args.n_head, self.args.d_head).transpose(1, 2) #<-nhead는 전체 문장의 길이, d-head는 dk를 의미하는 것 같은데, 왜 transpose를 하는 건지 이해가 안됨
        k_s = self.W_K(k).view(batch_size, -1, self.args.n_head, self.args.d_head).transpose(1, 2)
        v_s = self.W_V(v).view(batch_size, -1, self.args.n_head, self.args.d_head).transpose(1, 2)


        if attn_mask is not False:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_head, 1, 1)
            context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask) #context를 출력하고, attn_prob는 어디에 붙여서 쓰는 건지.
        else:
            context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.args.n_head * self.args.d_head)
        
        #<-encoder layer의 output dimension을 본래의 dimension으로 축소시키기 위한 FC
        output = self.linear(context)
        output = self.dropout(output)

        return output, attn_prob


class poswise_feedforward_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 주로 시계열데이터에 사용하는 convolution: 가로방향으로만 작동
        self.conv1 = nn.Conv1d(in_channels=self.args.d_hidn, out_channels=self.args.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.args.d_ff, out_channels=self.args.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, inputs):
        output = self.conv1(inputs.transpose(1, 2).contiguous())
        output = self.active(output)
        output = self.conv2(output).transpose(1, 2).contiguous()
        output = self.dropout(output)

        return output

class encoderlayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.self_attn = multihead_attention(self.args)
        self.pos_ffn = poswise_feedforward_net(self.args)

    def forward(self, inputs):
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs)
        att_outputs = att_outputs + inputs  # <- residual connection을 위한 코드로 보임.

        ffn_outputs = self.pos_ffn(att_outputs) # <- feed forward network의 결과값
        ffn_outputs = ffn_outputs + att_outputs # <-residual connection을 위한 코드

        return ffn_outputs, attn_prob


class encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enc_emb = nn.Linear(in_features=self.args.window_size, out_features=self.args.d_hidn, bias=False)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.args.e_features, self.args.d_hidn))
        ## sinusoid_table이 (2,512)형태로 입력값을 갖는다는 것을 확인할 수 있음. 

        # 공식문서 찾아보기(from_pretrained)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([encoderlayer(self.args) for _ in range(self.args.n_layer)]) #layer를 생성해주는 함수?
        self.enc_attn_probs = None

    def forward(self, inputs):  # inputs.shape = (batch_size, window_size, n_features)  ex) (128, 10 , 4) 왜 feature가 4개지?
        self.enc_attn_probs = []
   
        ## position코드는 진짜 뭘 의미하는지 모르겠다.
        positions = torch.arange(inputs.size(2), device=inputs.device).expand(inputs.size(0), inputs.size(2)).contiguous()  # (batch_size, n_features)  ex) (128, 4)
        outputs = self.enc_emb(inputs.transpose(2, 1).contiguous()) + self.pos_emb(positions)  # (batch_size, n_features, n_hidn)  ex) (128, 4, 256)

        for layer in self.layers:
            outputs, enc_attn_prob = layer(outputs)
            self.enc_attn_probs.append(enc_attn_prob)

        return outputs

    
class decoderlayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.self_attn = multihead_attention(self.args)
        self.dec_enc_attn = multihead_attention(self.args)
        self.pos_ffn = poswise_feedforward_net(self.args)

    def forward(self, dec_inputs, enc_outputs, attn_mask):
        self_att_outputs, dec_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, attn_mask)
        self_att_outputs = self_att_outputs + dec_inputs

        dec_enc_att_outputs, dec_enc_attn_prob = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs)
        dec_enc_att_outputs = dec_enc_att_outputs + self_att_outputs

        # cross attention: decoder에서, encoder의 output과 decoder input을 섞어주는 것. 
        
        ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
        ffn_outputs = ffn_outputs + dec_enc_att_outputs

        return ffn_outputs, dec_attn_prob, dec_enc_attn_prob



class decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dec_emb = nn.Linear(in_features=self.args.d_features, out_features=self.args.d_hidn, bias=False)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.args.window_size , self.args.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([decoderlayer(self.args) for _ in range(self.args.n_layer)])
        self.dec_attn_probs = None
        self.dec_enc_attn_probs = None

    def forward(self, dec_inputs, enc_outputs):
        self.dec_attn_probs = []  
        self.dec_enc_attn_probs = []
        positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous()
        dec_output = self.dec_emb(dec_inputs) + self.pos_emb(positions)
        
        # window size에 따라 뒷부분을 알수 있기 때문에, 이를 가려주기 위해 masking을 해주어야 함.
        attn_mask = torch.gt(get_attn_decoder_mask(dec_inputs), 0)

        for layer in self.layers:
            dec_outputs, dec_attn_prob, dec_enc_attn_prob = layer(dec_output, enc_outputs, attn_mask)
            self.dec_attn_probs.append(dec_attn_prob)
            self.dec_enc_attn_probs.append(dec_enc_attn_prob)
        
        return dec_outputs


class TimeDistributed(nn.Module): #<-이 부분이 어떤 코드인지 잘 모르겠음.
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):

        if len(x.size()) <= 2:  # (batch_size, window_size * d_hidn)  ex) (128, 2560)
            return self.module(x)
        
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)

        if len(x.size()) == 3:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))

        return y


class transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = encoder(self.args)
        self.decoder = decoder(self.args)
        self.fc1 = TimeDistributed(nn.Linear(in_features=self.args.window_size*self.args.d_hidn, out_features=self.args.dense_h))
        self.fc2 = TimeDistributed(nn.Linear(in_features=self.args.dense_h, out_features=self.args.ahead * self.args.output_size))
        self.activation = nn.SELU()

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)  # (batch_size, n_features, d_hidn)  ex) (128, 4, 256)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)  # (batch_size, window_size, d_hidn)  ex) (128, 10, 256)
        
        # dec_outputs.view(dec_outputs.size(0), -1).shape = (batch_size, window_size * d_hidn)  ex) (128, 2560)
        dec_outputs = self.fc1(dec_outputs.view(dec_outputs.size(0), -1))  # (batch_size, dense_h)   ex) (128, 128)
        dec_outputs = self.activation(dec_outputs)
        dec_outputs = self.fc2(dec_outputs)  # (batch_size, ahead * output_size)  ex) (128, 8)                
        dec_outputs = dec_outputs.view(dec_outputs.size(0), self.args.ahead, self.args.output_size)  #  ex) (128, 2, 4)

        return dec_outputs