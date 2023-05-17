import pandas as pd
from LSTM import Classifier

def main():
    
    path = 'all_EEG_data.csv'
    Time_df = pd.DataFrame(columns=['seed', 'Timestep', 'Acc', 'Val_acc', 'Loss', 'Val_loss'])


    for seed in range(5):
        df_ = Classifier(path, seed=seed, timesteps=256) 

    Time_df = Time_df.append(df_.return_results())

    print(Time_df)
    Time_df.to_csv('TimeStepDF_.csv')

if __name__ == "__main__":
    main()
