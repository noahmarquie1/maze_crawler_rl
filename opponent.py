def decision_tree_opponent(obs, config):
    actions = {}
    walls = obs.walls
    width = int(config.width)

    for uid, data in obs.robots.items():
        rtype, col, row, owner, jump_cd = (
            int(data[0]),
            int(data[1]),
            int(data[2]),
            data[4],
            int(data[6]),
        )

        if owner != obs.player:
            continue
        if rtype != 0:  # only factory acts
            continue
        if row >= obs.northBound:  # moving or jumping north would leave the board
            continue

        wall_index = (row - obs.southBound) * width + col
        wall_value = walls[wall_index]

        if not (wall_value & 1):
            actions[uid] = "NORTH"
        elif jump_cd == 0 and row + 2 <= obs.northBound:
            actions[uid] = "JUMP_NORTH"
        elif col > 0 and not (wall_value & 8):
            actions[uid] = "WEST"
        elif col < width - 1 and not (wall_value & 2):
            actions[uid] = "EAST"
        else:
            actions[uid] = "IDLE"

    return actions