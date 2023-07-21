# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from collections import deque


# This is your team name
CREATOR = "JustDontLoseLol"
VEHICLES = {"tanks", "ships", "jets"}
ALL_ENTITIES = {"bases", "tanks", "ships", "jets"}


# This is the AI bot that will be instantiated for the competition
class PlayerAi:
    def get_heading_to_nearest_enemy(self, position, enemy_info):
        """Returns the heading to the nearest enemy vehicle from the given position."""
        min_dist = float('inf')
        nearest_unit = None

        for team, team_info in enemy_info.items():
            for unit_type in VEHICLES:
                for unit in team_info.get(unit_type, []):
                    dist = np.linalg.norm(np.array([unit.x, unit.y]) - np.array(position))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_unit = unit

        if not nearest_unit:
            return 0  # No enemy vehicles found, return default heading

        # Calculate heading
        dx = nearest_unit.x - position[0]
        dy = nearest_unit.y - position[1]
        heading = np.degrees(np.arctan2(dy, dx)) % 360

        return heading

    def is_enemy_nearby(self, position, enemy_info):
        """Checks if there's a nearby enemy from the given position for any unit type."""

        # Define the proximity thresholds for each type of vehicle and base
        thresholds = {
            'tanks': 100,
            'jets': 150,
            'ships': 60,
        }

        for team, team_info in enemy_info.items():
            for unit_type, threshold in thresholds.items():
                for unit in team_info.get(unit_type, []):
                    dist = np.linalg.norm(np.array([unit.x, unit.y]) - np.array(position))
                    if dist < threshold:
                        return True
        return False

    def __init__(self):
        self.team = CREATOR  # Mandatory attribute

        # Record the previous positions of all my vehicles
        self.previous_positions = {}
        # Record the number of tanks and ships I have at each base
        self.ntanks = {}
        self.nships = {}
        self.njets = {}
        # Additional info
        self.OGBase = None
        self.OGBasePos = [None, None]
        self.start_time = None
        self.vehicle_step_counter = {}
        self.jet_steps = {}
        self.explore_queue = deque()


    def get_my_info(self, info):
        """Returns the information about my team."""
        return info[self.team]

    def get_enemy_info(self, info):
        """Returns the information about enemy teams."""
        enemy_info = {}
        for team, team_info in info.items():
            if team != self.team:
                enemy_info[team] = team_info
        return enemy_info

    def nearest_enemy_base(self, vehicle, enemy_info):
        """
        Returns the coordinates of the nearest enemy base to the vehicle.
        If no enemy bases are found, return None.
        """
        nearest_base = None
        min_distance = float('inf')

        for team, team_info in enemy_info.items():
            for base in team_info.get("bases", []):
                distance = ((vehicle.x - base["x"]) ** 2 + (vehicle.y - base["y"]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_base = (base["x"], base["y"])

        return nearest_base

    def is_enemy_nearby(self, position, enemy_info):
        """Checks if there's a nearby enemy from the given position for any unit type."""

        # Define the proximity thresholds for each type of vehicle and base
        thresholds = {
            'tanks': 100,
            'jets': 150,
            'ships': 60,
        }

        for team, team_info in enemy_info.items():
            for unit_type, threshold in thresholds.items():
                for unit in team_info.get(unit_type, []):
                    dist = np.linalg.norm(np.array([unit.x, unit.y]) - np.array(position))
                    if dist < threshold:
                        return True
        return False


    def base_behavior(self, base, t, game_map, myinfo, enemy_info):
        # If this is a new base, initialize the tank & ship counters
        if base.uid not in self.ntanks:
            self.ntanks[base.uid] = 0
        if base.uid not in self.nships:
            self.nships[base.uid] = 0
        if base.uid not in self.njets:
            self.njets[base.uid] = 0

        total_number_of_my_bases = len(myinfo["bases"])

        # First, always check if there's a threat nearby and spawn a tank for defense
        enemy_nearby = self.is_enemy_nearby([base.x, base.y], enemy_info)
        # If enemy nearby, create defensive tanks
        if enemy_nearby:
            max_iterations = 5
            i = 0
            while base.crystal > base.cost("tank") and i < max_iterations:
                if self.ntanks[base.uid] > 20:
                    i += 5
                    continue
                direction_to_enemy = self.get_heading_to_nearest_enemy([base.x, base.y], enemy_info)
                base.build_tank(heading=direction_to_enemy)
                self.ntanks[base.uid] += 1
                i += 1

        # Firstly, each base should build a mine if it has less than 3 mines
        if base.mines < 3:
            if base.crystal > base.cost("mine"):
                base.build_mine()

        # Secondly, each base should build a tank if it has less than 5 tanks
        elif base.crystal > base.cost("tank") and self.ntanks[base.uid] < 5:
            # build_tank() returns the uid of the tank that was built
            tank_uid = base.build_tank(heading=360 * np.random.random())
            # Add 1 to the tank counter for this base
            self.ntanks[base.uid] += 1
        # Thirdly, each base should build a ship if it has less than 3 ships
        elif base.crystal > base.cost("ship") and self.nships[base.uid] < 3:
            # build_ship() returns the uid of the ship that was built
            ship_uid = base.build_ship(heading=360 * np.random.random())
            # Add 1 to the ship counter for this base
            self.nships[base.uid] += 1
        elif base.crystal > base.cost("ship") and total_number_of_my_bases < 5 and self.nships[base.uid] < 15:
            ship_uid = base.build_ship(heading=360 * np.random.random())
            self.nships[base.uid] += 1
        elif base.crystal > base.cost("tank") and self.njets[base.uid] > 5 and self.ntanks[base.uid] < 20:
            tank_uid = base.build_tank(heading=360 * np.random.random())
            # Add 1 to the tank counter for this base
            self.ntanks[base.uid] += 1
        # If everything else is satisfied, build a jet
        elif base.crystal > base.cost("jet"):
            # build_jet() returns the uid of the jet that was built
            jet_uid = base.build_jet(heading=360 * np.random.random())
            self.njets[base.uid] += 1


    def run(self, t: float, dt: float, info: dict, game_map: np.ndarray):
        myinfo = self.get_my_info(info)
        enemy_info = self.get_enemy_info(info)

        # First run definitions
        if self.start_time is None:
            self.start_time = t
        if not self.OGBase:
            self.OGBase = myinfo["bases"][0]
            self.OGBasePos = [self.OGBase.x, self.OGBase.y]

        # Behavior for bases
        for base in myinfo["bases"]:
            self.base_behavior(base, t, game_map, myinfo, enemy_info)

        # Vehicle step counter for optimization
        for vehicle_str in VEHICLES:
            if vehicle_str not in myinfo:
                continue
            for vehicle in myinfo[vehicle_str]:
                if vehicle.uid not in self.vehicle_step_counter:
                    self.vehicle_step_counter[vehicle.uid] = 0
                self.vehicle_step_counter[vehicle.uid] += 1

        # Behavior for tanks
        if "tanks" in myinfo:
            for tank in myinfo["tanks"]:
                if self.vehicle_step_counter[tank.uid] % 10 != 0:
                    continue
                self.tank_behavior(tank, game_map, myinfo, enemy_info)

        # Behavior for ships
        if "ships" in myinfo:
            for ship in myinfo["ships"]:
                if self.vehicle_step_counter[ship.uid] % 10 != 0:
                    continue
                self.ship_behavior(ship, game_map, enemy_info)

        # Behavior for jets
        if "jets" in myinfo:
            for jet in myinfo["jets"]:
                if self.vehicle_step_counter[jet.uid] % 10 != 0:
                    continue
                self.jet_behavior(jet, game_map, enemy_info)


    def tank_behavior(self, tank, game_map, myinfo, enemy_info):
        target = None

        # 1. Compute the nearest friendly base's coordinates
        nearest_base_distance = float('inf')
        nearest_base = None

        for base in myinfo["bases"]:
            distance = ((tank.x - base["x"]) ** 2 + (tank.y - base["y"]) ** 2) ** 0.5
            if distance < nearest_base_distance:
                nearest_base_distance = distance
                nearest_base = (base["x"], base["y"])

        # if too far, go back
        if nearest_base_distance > 120:
            target = nearest_base

        # 2. Prioritize killing enemies within a 100px radius
        nearest_enemy_distance = float('inf')
        nearest_enemy = None

        for team, team_info in enemy_info.items():
            for entity_type in ["tanks", "jets"]:
                for enemy_entity in team_info.get(entity_type, []):
                    distance = ((tank.x - enemy_entity["x"]) ** 2 + (tank.y - enemy_entity["y"]) ** 2) ** 0.5
                    if distance <= 100:
                        nearest_enemy_distance = distance
                        nearest_enemy = (enemy_entity["x"], enemy_entity["y"])

        if nearest_enemy:
            target = nearest_enemy

        if (tank.uid in self.previous_positions) and (not tank.stopped):
            # If the tank position is the same as the previous position,
            # set a random heading
            if target:
                tank.goto(*target)
            elif all(tank.position == self.previous_positions[tank.uid]):
                tank.set_heading(np.random.random() * 360.0)
        self.previous_positions[tank.uid] = tank.position

    def ship_behavior(self, ship, game_map, enemy_info):
        if ship.uid in self.previous_positions:
            # If the ship position is the same as the previous position,
            # convert the ship to a base if it is far from the owning base,
            # set a random heading otherwise
            if all(ship.position == self.previous_positions[ship.uid]):
                if ship.get_distance(ship.owner.x, ship.owner.y) > 40:
                    ship.convert_to_base()
                else:
                    ship.set_heading(np.random.random() * 360.0)
        # Store the previous position of this ship for the next time step
        self.previous_positions[ship.uid] = ship.position

    def jet_behavior(self, jet, game_map, enemy_info):
        # 1. Attack enemy bases if visible
        nearest_base = self.nearest_enemy_base(jet, enemy_info)
        if nearest_base:
            jet.goto(*nearest_base)
            return

        # If jet is not in the dictionary, add it
        if jet.uid not in self.jet_steps:
            self.jet_steps[jet.uid] = 1

        if self.vehicle_step_counter[jet.uid] % (20 * self.jet_steps[jet.uid]) == 0:
            # random number between 50 and 90
            random_angle = np.random.random() * 40 + 50
            new_heading = (jet.heading + random_angle) % 360  # Make sure heading is between 0 and 360
            # Increase the step counter
            self.jet_steps[jet.uid] += 1
            self.vehicle_step_counter[jet.uid] = 0
            jet.set_heading(new_heading)
