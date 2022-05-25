import argparse
import easydict
import numpy as np
import pandas as pd
from datetime import timedelta

import torch
from utils.preprocessor import csv_to_pd
from utils.plots import plot_inference_result
from models.transformer import transformer


def inference(opt):
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    device = 'cpu'

    with open(opt.weight, 'rb') as f:
        saved_model = torch.load(f)
    scaler = saved_model['scaler']
    window_size = saved_model['window_size']
    ahead = saved_model['ahead']
    print(f"best epoch: {saved_model['best_epoch']}")
    print(f"window_size: {window_size}")
    print(f'ahead: {ahead}')

    observed_df = csv_to_pd(opt.test_data)   

    # window_size까지만 추출하고,
    last_window_days = observed_df[-window_size:]   

    model_args = easydict.EasyDict({
        'output_size': saved_model['output_size'],
        'window_size': window_size,
        'ahead': ahead,
        'batch_size': 1,        
        'e_features': saved_model['e_features'],
        'd_features': saved_model['d_features'],
        'd_hidn': saved_model['d_hidn'],
        'n_head': saved_model['n_head'],
        'd_head': saved_model['d_head'],
        'dropout': saved_model['dropout'],
        'd_ff': saved_model['d_ff'],
        'n_layer': saved_model['n_layer'],
        'dense_h': saved_model['dense_h'],        
        'device': device
    })

    # 모델을 선언하고,
    model = transformer(model_args).to(device)

    # 아마도, saved_model['state']에 저장된 파라미터의 값을 그대로 가져와 모델에 로딩하는 코드로 보임.
    model.load_state_dict(saved_model['state'])    
    
    # 이건 무슨 코드지.
    model.eval()    

    with torch.no_grad():
        # 최근 14일간 날짜를 numpy 배열로 만들고 (.to_numpy),
        # 만들어진 14,2(2개 국가이므로)를 1열의 flatten된 배열로 만들어주는 코드
        x = last_window_days.to_numpy().reshape(-1)

        # minmax scaler를 통해 너무 큰 값을 잡아줌 (매우 큰 값은 이상치라는 의미인가?)
        x = scaler.transform(np.expand_dims(x, axis=1))

        #왜 (2,14?)가 되지? 이해가 안되네.
        x = x.reshape(-1, observed_df.shape[-1])        
        x = torch.from_numpy(x).float().unsqueeze(0) # 0가 있는 경우에도 squeeze하지 않는다는 의미같은데,
        y_pred = model(x.to(device), x.to(device))   # y_pred.shape = (batch_size, ahead. n_features)  ex) (1, 2, 4)            
        y_pred = y_pred.cpu().numpy()        

    prediction = []    
    for idx, col in enumerate(observed_df.columns):                
        future = pd.date_range(observed_df.index[-1], periods=ahead + 1)
        predicted_cases = scaler.inverse_transform(y_pred[:, :, idx]).flatten()        
        print(predicted_cases)
        prediction.append(predicted_cases)
        plot_inference_result(observed_df, col, future, np.append(observed_df[col][-1], predicted_cases))  # append the last observed day to draw a prettier graph
        
    prediction = np.stack(prediction)
    return prediction, ahead


def write_down(prediction, ahead, pred_csv):

    df_form = csv_to_pd(opt.test_data)    
    columns = [x for x in df_form.columns]

    yesterday = df_form.index[-1]
    today = yesterday + timedelta(days=1)
    future = yesterday + timedelta(days=ahead)
    time_series = pd.date_range(today, periods=(future - yesterday).days)
    
    pred_df = pd.DataFrame(index=time_series, columns=columns)
    pred_df.index.name = 'timestamp'    
    
    for t, time_point in enumerate(time_series):
        for c, col in enumerate(pred_df.columns):
            pred_df.loc[time_point][col] = prediction[c][t]
    
    pred_df.to_csv(pred_csv, sep='\t')



if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, help='path to the test data')        
    parser.add_argument('--weight', type=str, help='path to the weight file')    
    parser.add_argument('--pred_csv', type=str, help='path to the prediction output')
    opt = parser.parse_args()
    
    prediction, ahead = inference(opt)
    write_down(prediction, ahead, opt.pred_csv)