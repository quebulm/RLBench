from typing import List
from rlbench.backend.task import Task

from rlbench.const import colors
from rlbench.backend.task_utils import sample_procedural_objects
from rlbench.backend.conditions import ConditionSet, DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy

class SortingChallenge(Task):

    def init_task(self) -> None:
        # This is called once when a task is initialised.
        self.target = Shape('Target')
        self.spawn_boundary = Shape('spawn_boundary')
        self.success_detectors = [ProximitySensor(f'success{i}') for i in range(4)]
        self.waypoints = [Dummy(f'waypoint{i}') for i in range(4)]

        self.object_circle = Shape('Circle')
        self.object_cube = Shape('Cube')
        self.object_rectangle = Shape('Rectangle')
        self.object_triangle = Shape('Triangle')

        self.register_waypoint_ability_start(1, self._move_above_object)
        self.register_waypoints_should_repeat(self._repeat)

    def _move_above_object(self, waypoint):
        if len(self.bin_objects_not_done) <= 0:
            raise RuntimeError('Should not be here.')
        next_obj = self.bin_objects_not_done[0]  # Nächstes zu bewegendes Objekt
        x, y, z = next_obj.get_position()  # Position des Objekts
        waypoint.get_waypoint_object().set_position([x, y, z])  # Wegpunkt aktualisieren

    def _repeat(self):
        return len(self.bin_objects_not_done) > 0

    def init_episode(self, index: int) -> List[str]:
        # Spawn objects and recognize colors
        self.spawned_objects = [self.object_circle, self.object_cube,
                                self.object_rectangle, self.object_triangle]
        self.bin_objects_not_done = list(self.spawned_objects)
        # Assign objects to proximity sensors
        self.object_to_sensor = {
            self.object_circle: self.success_detectors[2],
            self.object_triangle: self.success_detectors[1],
            self.object_rectangle: self.success_detectors[3],
            self.object_cube: self.success_detectors[0]
        }

        # Select colors for the objects
        target_color_name, target_color_rgb = colors[index]
        color_indices = list(range(len(colors)))
        color_indices.remove(index)
        for obj in self.spawned_objects:
            color_choice = np.random.choice(color_indices)
            _, color_rgb = colors[color_choice]
            obj.set_color(color_rgb)
            color_indices.remove(color_choice)  # Ensure different colors for each object

        # Success conditions
        conditions = []
        for obj, sensor in self.object_to_sensor.items():
            conditions.append(DetectedCondition(obj, sensor))

        # Set the success conditions
        self.register_success_conditions(
            [ConditionSet(conditions, simultaneously_met=True)])

        # place objects
        b = SpawnBoundary(list(self.spawned_objects))
        for ob in self.spawned_objects:
            ob.set_position([0.0, 0.0, 0.2], relative_to=self.target, reset_dynamics=False)
            b.sample(ob, ignore_collisions=True, min_distance=0.05)

        # set waypoints to spawned objects
        for obj in self.spawned_objects:
            self.set_pickup_waypoints(obj)
            sensor = self.object_to_sensor[obj]
            self.set_dropoff_waypoint(sensor)

        return ['Place the objects in the correct slots']


    def set_dropoff_waypoint(self, sensor):
        dropoff_pos = sensor.get_position()
        self.waypoints[3].set_position(dropoff_pos)

    def set_pickup_waypoints(self, object):
        pickup_pos = object.get_position()
        self.waypoints[0].set_position(pickup_pos)
        self.waypoints[1].set_position(pickup_pos)

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def step(self) -> None:
        # Called during each sim step. Remove this if not using.
        # iterate the not placed objects
        for ob in self.bin_objects_not_done[:]:
            sensor = self.object_to_sensor[ob]  # get sensor
            if sensor.is_detected(ob):
                self.bin_objects_not_done.remove(ob)  # remove if success

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        for obj in self.spawned_objects:
            obj.remove()
