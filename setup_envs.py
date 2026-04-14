from experiments import main as evaluate_heuristic_agents

if __name__ == "__main__":
    #env_name = "l2rpn_case14_sandbox"
    #env = grid2op.make(env_name)
    #env.train_val_split_random(add_for_train="train", add_for_test="test", add_for_val="val", pct_val=5., pct_test=5.)

    #env_name = "l2rpn_wcci_2020"
    #env = grid2op.make(env_name)
    #env.train_val_split_random(add_for_train="train", add_for_test="test", add_for_val="val", pct_val=5., pct_test=5.)

    #env_name = "l2rpn_neurips_2020_track2_large"
    #env = grid2op.make(env_name)
    #env.train_val_split_random(add_for_train="train", add_for_test="test", add_for_val="val", pct_val=5., pct_test=5.)

    evaluate_heuristic_agents()