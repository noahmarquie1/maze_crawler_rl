Crawl: Getting Started

This guide walks you through building an agent, testing it locally, and submitting it to the Crawl competition on Kaggle.
Game Overview

Crawl is a two-player real-time strategy game on a 20-wide maze that scrolls northward over time. Each player starts with a single Factory and must build robots to explore, collect energy, and outlast the opponent.

    Factory (indestructible) builds Scouts, Workers, and Miners; can JUMP every 20 turns
    Scout (cost 50) is fast with vision range 5 — your eyes
    Worker (cost 200) builds and removes walls (100 energy per action)
    Miner (cost 300) can TRANSFORM on a mining node into a mine that generates 50 energy/turn
    Maze has east/west symmetry with occasional doors; both players see only what their robots see (fog of war)
    Combat: when robots end the turn on the same cell, crush rules apply (Factory > Miner > Worker > Scout). Same-type collisions destroy all parties — friendly fire is real
    Scrolling: the southern boundary advances, destroying anything left behind. Speed ramps from 1/4 turns to 1/turn by step 400
    Win condition: last factory standing wins; if both survive to step 500, tiebreaker cascade is total energy → unit count → draw

See How to Play Maze Crawler for full rules and configuration defaults.
Your Agent

Your agent is a function that receives an observation and configuration and returns a dict mapping robot UIDs to action strings.

Observation fields:

    obs.player — your player index (0 or 1)
    obs.walls — flat array of wall bitfields. Index = (row - southBound) * width + col. Value -1 = undiscovered. Bits: N=1, E=2, S=4, W=8
    obs.crystals — {"col,row": energy}, only currently visible
    obs.robots — {"uid": [type, col, row, energy, owner, move_cd, jump_cd, build_cd]}. Types: 0=Factory, 1=Scout, 2=Worker, 3=Miner
    obs.mines — {"col,row": [energy, maxEnergy, owner]}, remembered once seen
    obs.miningNodes — {"col,row": 1}, only currently visible
    obs.southBound, obs.northBound — current active row range

Action format: Each value is an action string keyed by robot UID:

    Movement: NORTH, SOUTH, EAST, WEST, IDLE
    Factory: BUILD_SCOUT, BUILD_WORKER, BUILD_MINER, JUMP_NORTH/SOUTH/EAST/WEST
    Worker: BUILD_NORTH/SOUTH/EAST/WEST, REMOVE_NORTH/SOUTH/EAST/WEST
    Miner: TRANSFORM (must be on a mining node)
    Any robot: TRANSFER_NORTH/SOUTH/EAST/WEST to send all energy to an adjacent friendly
