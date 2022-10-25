import wandb

api = wandb.Api()
sweep_id = "impala/z1mlw147"
sweep = api.sweep(sweep_id)
print(sweep)

# with open("sweep.yaml", "r") as f:
#    sweep_config = yaml.safe_load(f)
# sweep_config["name"] = "test"
# out = wandb.sweep(sweep_config, project="test")
# print(out)
