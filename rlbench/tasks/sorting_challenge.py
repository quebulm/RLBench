from typing import List, Tuple
from rlbench.backend.task import Task

from rlbench.const import colors
from rlbench.backend.task_utils import sample_procedural_objects
from rlbench.backend.conditions import ConditionSet, DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy

import random

class SortingChallenge(Task):

    def init_task(self) -> None:
        # This is called once when a task is initialised.
        self.target = Shape('Target')
        self.spawn_boundaries = [Shape('spawn_boundary')]
        self.success_detectors = [ProximitySensor(f'success{i}') for i in range(4)]
        self.waypoints = [Dummy(f'waypoint{i}') for i in range(5)]

        self.object_circle = Shape('Circle')
        self.object_cube = Shape('Cube')
        self.object_rectangle = Shape('Rectangle')
        self.object_triangle = Shape('Triangle')

        self.initial_z = 0.77  # Z position  (because no parent)

        graspable_objects = [self.object_circle, self.object_cube, self.object_rectangle, self.object_triangle]

        # register objects graspable for the panda
        self.register_graspable_objects(graspable_objects)

    def _move_above_object(self, waypoint):
        if len(self.bin_objects_not_done) <= 0:
            raise RuntimeError('Should not be here.')
        x, y, z = self.bin_objects_not_done[0].get_position()
        waypoint.get_waypoint_object().set_position([x, y, z])

    def _repeat(self):
        return len(self.bin_objects_not_done) > 0

    def init_episode(self, index: int) -> List[str]:
        # Spawn objects and recognize colors
        self.spawned_objects = [self.object_circle, self.object_cube,
                                self.object_rectangle, self.object_triangle]
        self.bin_objects_not_done = list(self.spawned_objects)
        # Assign objects to proximity sensors
        self.object_to_sensor = {
            self.object_circle.get_name(): self.success_detectors[2],
            self.object_triangle.get_name(): self.success_detectors[1],
            self.object_rectangle.get_name(): self.success_detectors[3],
            self.object_cube.get_name(): self.success_detectors[0]
        }

        # Select colors for the objects
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
            obj_instance = self.get_object_by_name(obj)
            conditions.append(DetectedCondition(obj_instance, sensor))

        # Set the success conditions
        self.register_success_conditions(
            [ConditionSet(conditions, simultaneously_met=True)])

        # # Platzieren der Objekte
        b = SpawnBoundary(self.spawn_boundaries)
        for ob in self.spawned_objects:
            # Speichern der aktuellen Rotation des Objekts
            current_orientation = ob.get_orientation()

            # Setzen der neuen Position
            ob.set_position([0.0, 0.0, self.initial_z], relative_to=self.target, reset_dynamics=False)
            b.sample(ob, ignore_collisions=False, min_distance=0.1)

            # Abrufen der neuen X- und Y-Position und Zurücksetzen auf die ursprüngliche Z-Position
            x, y, _ = ob.get_position()
            ob.set_position([x, y, self.initial_z])

            # Wiederherstellen der ursprünglichen Rotation des Objekts
            ob.set_orientation(current_orientation)

        selected_object = random.choice(self.spawned_objects)
        print(f"selected Object: {selected_object.get_name()}")
        selected_object_pos = selected_object.get_position()
        print(f"Position von Object: {selected_object_pos}")
        self.set_pickup_waypoints(selected_object)
        selected_object_sensor = self.object_to_sensor[selected_object.get_name()]
        self.set_dropoff_waypoint(selected_object_sensor)

        return ['Place the objects in the correct slots']


    def set_dropoff_waypoint(self, sensor):
        dropoff_pos = sensor.get_position()
        self.waypoints[3].set_position([dropoff_pos[0], dropoff_pos[1], self.initial_z+0.2], relative_to=None,
                                       reset_dynamics=False)

        self.waypoints[4].set_position([dropoff_pos[0], dropoff_pos[1], self.initial_z+0.05], relative_to=None,
                                       reset_dynamics=False)

        # Debug-Ausgabe für die Position von Wegpunkt 3
        waypoint_3_pos = self.waypoints[3].get_position()
        print(f"Position von Wegpunkt 3: {waypoint_3_pos}")

    def set_pickup_waypoints(self, wobject):
        pickup_pos = wobject.get_position()

        self.waypoints[0].set_position([pickup_pos[0], pickup_pos[1], self.initial_z+0.2], relative_to=None,
                                       reset_dynamics=False)

        self.waypoints[1].set_position([pickup_pos[0], pickup_pos[1], self.initial_z], relative_to=None,
                                       reset_dynamics=False)


        # Debug-Ausgabe für die Position von Wegpunkt 1
        waypoint_0_pos = self.waypoints[0].get_position()
        print(f"Position von Wegpunkt 0: {waypoint_0_pos}")

        # Debug-Ausgabe für die Position von Wegpunkt 1
        waypoint_1_pos = self.waypoints[1].get_position()
        print(f"Position von Wegpunkt 1: {waypoint_1_pos}")


    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def step(self) -> None:
        # Called during each sim step. Remove this if not using.
        # iterate the not placed objects
        # Wird bei jedem Simulationsschritt aufgerufen.
        for ob in self.bin_objects_not_done[:]:
            sensor = self.object_to_sensor[ob.get_name()]
            if sensor.is_detected(ob):
                self.bin_objects_not_done.remove(ob)
                # TODO: Setzen Sie hier die neuen Wegpunkte, wenn das Objekt erkannt wurde.
               # self.set_pickup_waypoints(ob)
               # self.set_dropoff_waypoint(sensor)

    def cleanup(self) -> None:

        # Called during at the end of each episode. Remove this if not using.
        pass

    def get_object_by_name(self, name):
        if name == self.object_circle.get_name():
            return self.object_circle
        elif name == self.object_cube.get_name():
            return self.object_cube
        elif name == self.object_rectangle.get_name():
            return self.object_rectangle
        elif name == self.object_triangle.get_name():
            return self.object_triangle
        else:
            raise ValueError("Unbekanntes Objekt: " + name)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]