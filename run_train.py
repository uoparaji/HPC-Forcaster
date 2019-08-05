#"training_set_size=1841_weeks=7.csv"
#"training_set_size=7556_weeks=17.csv"
#"training_set_size=37568_weeks=43.csv"
#"training_set_size=150076_weeks=80.csv"
def run_train():
    from train import train_lstm
    train_lstm("training_set_size=37568_weeks=43.csv", lag=23, optim='rmsprop', epo=17, layer1 = 227, layer2 = 97)