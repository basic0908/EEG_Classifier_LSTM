# Base vs Target binary classification(LSTM)

ベストモデルの判別精度62%だが、何度か回して出た結果なので有意な精度とは言えない

Preprocess:
- データを[1000:8680]の範囲(15秒程度)でスライス
- all_data.csvのように整理
- 外れ値処理、正規化など試す

Model:
- LSTM 15 -> Dense(50) -> Pooling() -> Dense(5) -> Dense(5)
- https://digitalcommons.georgiasouthern.edu/cgi/viewcontent.cgi?article=3626&context=etd
- LSTM_EpilepticSeizureRecognition.pdf
- https://gist.github.com/urigoren/b7cd138903fe86ec027e715d493451b4
などを参考にしたが精度は上がらず

