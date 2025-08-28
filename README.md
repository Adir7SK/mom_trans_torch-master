
Create training run settings by creating yaml in "mom_trans_torch/configs/train_settings" e.g. "pinnacle-gross-futs-and-fx.yaml"

You will need to reference your wandby account name in "mom_trans_torch/configs/settings.py" by setting the `WANDB_ENTITY` variable. Then, in wandb tour wandb account setup a wandb project called `dmn` (since I used `project: dmn` in the yaml file)

Train the model ensemble with:
```
python -m mom_trans_torch.data.jobs.tune_hyperparameters --run_file_name "pinnacle-gross-futs-and-fx"  --arch LSTM_SIMPLE
```

Then backtest the relvant file by setting up a backtest setting file in "mom_trans_torch/configs/backtest_settings" similar to "lstm-simple-pinnacle-gross-futs-and-fx.yaml", which you then run with
```
python -m mom_trans_torch.backtest.jobs.make_predictions --name "lstm-simple-pinnacle-gross-futs-and-fx" 
```


