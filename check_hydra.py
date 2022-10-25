import hydra



import omegaconf
@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    out = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    print(out)
    #slurm_jobid = int(os.environ["SLURM_JOBID"][-4:])
    #os.environ['MASTER_PORT'] = str(10000 + slurm_jobid)



    print(cfg)

    pass
if __name__=="__main__":
    main()