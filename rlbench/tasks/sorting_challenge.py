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
        self.object_pentagon = Shape('Pentagon')
        self.object_rectangle = Shape('Rectangle')
        self.object_triangle = Shape('Triangle')

        self.register_waypoint_ability_start(1, self._move_above_object)
        self.register_waypoints_should_repeat(self._repeat)


    def init_episode(self, index: int) -> List[str]:
        # This is called at the start of each episode.
        # Objekte, die platziert werden sollen
        self.spawned_objects = [self.object_circle, self.object_cube, self.object_pentagon,
                                self.object_rectangle, self.object_triangle]
        for obj in self.spawned_objects:
            self.spawn_boundary.sample(obj)  # Objekt im Spawn-Bereich platzieren
        return ['Place the objects in the correct slots']


    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def step(self) -> None:
        # Called during each sim step. Remove this if not using.
        pass

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        for obj in self.spawned_objects:
            obj.remove()
