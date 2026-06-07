def game_agent(obs, fac_action):
    actions = {}
    for uid, data in obs.robots.items():
        rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
        if owner != obs.player:
            continue
        if rtype == 0:  # Factory
            actions[uid] = fac_action
        else:
            actions[uid] = "NORTH"
    return actions



