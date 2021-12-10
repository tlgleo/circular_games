from gym.envs.registration import register

register(
    id='circular_collect_special_4x4-v0',
    entry_point='circular_collect.envs:CircularCollect4players_4x4_4P_Cir',
)

register(
    id='circular_collect_special_4x4-v1',
    entry_point='circular_collect.envs:CircularCollect4players_4x4_4P_Bil',
)

register(
    id='circular_collect_special_4x4-v2',
    entry_point='circular_collect.envs:CircularCollect4players_4x4_4P_Uni_2',
)

register(
    id='circular_collect_special_4x4_tr-v0',
    entry_point='circular_collect.envs:CircularCollect4players_4x4_4P_Uni',
)







