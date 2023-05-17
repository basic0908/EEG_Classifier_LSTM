# Base vs Target binary classification(LSTM)

ベストモデルの判別精度62%だが、何度か回して出た結果なので有意な精度とは言えない

Preprocess:
- データを[1000:9680]の範囲(15秒程度)でスライス
- Left Row, Right Rowの平均値を算出
- all_data.csvのように整理
- 外れ値処理、正規化など試す

Model:
- LSTM 15 -> Dense(50) -> Pooling() -> Dense(5) -> Dense(5)
- https://digitalcommons.georgiasouthern.edu/cgi/viewcontent.cgi?article=3626&context=etd
- Epileptic Seizure Recognition using LSTM
- https://gist.github.com/urigoren/b7cd138903fe86ec027e715d493451b4
などを参考にしたが精度は上がらず

トレーニング例

![image](https://github.com/basic0908/EEG_Classifier_LSTM/assets/100826336/f6d7d00c-c8b0-4691-87d7-547aa4709be7)
![image](https://github.com/basic0908/EEG_Classifier_LSTM/assets/100826336/aca3cc84-4853-4817-87a4-251d83a91145)

Accuracyにそれほど向上が見えない、Confusion Matrixが散らばっているので

- ノイズ
- 有意性のないデータ
- モデルのアーキテクチャ
などが考えられる
