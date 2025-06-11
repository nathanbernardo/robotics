from asyncio import Lock
import json
import logging
from typing import Awaitable, Dict, Callable, Optional
from nats.js import JetStreamContext
import numpy as np
import numpy.typing as npt
import ikpy.chain

from enum import Enum, EnumMeta

from libs.ik_utils.generate_rotational_matrix import generate_rotational_matrix

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MetaEnum(EnumMeta):
    def __contains__(cls, item) -> bool:
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    pass


class State(Enum):
    STOPPED = "Stopped"
    CALIBRATING = "Calibrating"
    CALIBRATED = "Calibrated"
    PRESSING_POWER_BUTTON = "Pressing power button"
    PRESSED_POWER_BUTTON = "Pressed power button"
    GOING_HOME = "Going home"
    BACK_AT_HOME = "Back at home"
    LIFTING_WATER_HANDLE = "Lifting water handle"
    LIFTED_WATER_HANDLE_OFF = "Lifted water handle off"
    PICKING_UP_WATER_BOTTLE = "Picking up water bottle"
    GRABBED_WATER_BOTTLE = "Grabbed water bottle"
    MOVING_TO_WATER_BUCKET = "Moving to water bucket"
    ARRIVED_AT_WATER_BUCKET = "Arrived at water bucket"
    POURING_WATER = "Pouring water"
    POURED_WATER = "Poured water"
    YEET_WATER_BOTTLE = "Yeet water bottle"
    LIFTING_COFFEE_HANDLE = "Lifting coffee handle"
    LIFTED_COFFEE_HANDLE = "Lifted coffee handle"
    GRABBING_COFFEE_POD = "Grabbing coffee pod"
    GRABBED_COFFEE_POD = "Grabbed coffee pod"
    INSERTING_COFFEE_POD = "Inserting coffee pod"
    INSERTED_COFFEE_POD = "Inserted coffee pod"
    CLOSING_COFFEE_HANDLE = "Closing coffee handle"
    CLOSED_COFFEE_HANDLE = "Closed coffee handle"
    MOVING_TO_COFFEE_CUP = "Moving to coffee cup"
    GRABBING_COFFEE_CUP = "Grabbing coffee cup"
    GRABBED_COFFEE_CUP = "Grabbed coffee cup"
    PLACING_CUP_ON_PLATFORM = "Placing cup on platform"
    PLACED_CUP_ON_PLATFORM = "Placed cup on platform"
    PRESSING_8_OZ_BUTTON = "Presing 8oz button"
    PRESSED_COFFEE_BUTTON = "Pressed coffee button"
    WAITING_FOR_COFFEE_TO_FINISH = "Waiting for coffee to finish"
    DONE_WAITING = "Done waiting"
    PLACING_CUP_TO_USER = "Placing cup to user"
    PLACED_CUP_TO_USER = "Placed cup to user"
    DONE = "Done"


class KeurigObjects(BaseEnum):
    FOUR_OZ = "4oz"
    SIZ_OZ = "6oz"
    EIGHT_OZ = "8oz"
    TEN_OZ = "10oz"
    TWELVE_OZ = "12oz"
    POWER_BUTTON = "power"
    STRONG = "strong"
    ICED = "iced"
    HOT = "hot"
    ADD_WATER = "add_water"
    DESCALE = "descale"
    WATER_BIN = "water_bin"
    WATER_HANDLE = "water_handle"
    COFFEE_HANDLE = "coffee_handle"
    PLATFORM = "platform"
    POD_HOLDER = "pod_holder"
    COFFEE_CUP = "coffee_cup"
    POWER_BUTTON_ON = "power_on"
    KEURIG = "keurig"
    END_EFFECTOR = "end_effector"


keurig_labels = ["12oz", "keurig", "power_on", "water_handle", "platform"]

# print("true if string in Enum : ", "7oz" in KeurigObjects)


class StateMachine:
    def __init__(self, js: Optional[JetStreamContext]):
        self.state: State = State.CALIBRATING  # Initial state
        self.states: Dict[State, Callable[[], Awaitable[None]]] = {
            State.CALIBRATING: self.calibrating,
            State.CALIBRATED: self.calibrated,
            State.PRESSING_POWER_BUTTON: self.pressing_power_button,
            State.PRESSED_POWER_BUTTON: self.pressed_power_button,
            State.GOING_HOME: self.going_home,
            State.BACK_AT_HOME: self.back_at_home,
            State.STOPPED: self.stop,
        }
        self.prev_state: State = State.STOPPED
        self.lock: Lock = Lock()
        self.object_coord_camera_frame: Dict[str, npt.NDArray[np.float64]] = {}
        self.object_coord_robot_frame: Dict[str, npt.NDArray[np.float64]] = {}
        self.mask: int = 0
        self.transform: npt.NDArray[np.float64] = np.identity(4)
        self.robot_chain = ikpy.chain.Chain.from_urdf_file(
            "./packages/urdf/chungus.URDF"
        )
        self.js: Optional[JetStreamContext] = js
        self.cache_object_coord_data: Dict[str, npt.NDArray[np.float64]] = {}

    def get_current_state(self):
        print("[State Machine] Current state: ", self.state)
        return self.state

    def ask_for_coord_data(
        self, coord_data_camera_frame: Dict[str, npt.NDArray[np.float64]]
    ) -> None:
        if not isinstance(coord_data_camera_frame, dict):
            raise ValueError("Coordinate data must be a dictionary")
        for key, value in coord_data_camera_frame.items():
            if not isinstance(key, str) or key not in KeurigObjects:
                raise ValueError(f"Invalid object key: {key}")
            if not isinstance(value, np.ndarray) or value.shape != (3,):
                raise ValueError(
                    f"Invalid coordinate format for {key}: execpted 3x1 numpy array"
                )
        logger.info(f"Coord data in camera frame: {coord_data_camera_frame}")
        self.object_coord_camera_frame = coord_data_camera_frame

    def __transform_coordinates(
        self, camera_frame_coords: Dict[str, npt.NDArray[np.float64]]
    ) -> Dict[str, npt.NDArray[np.float64]]:
        """Transform  coordinates from camera frame to robot frame"""
        matrices = list(camera_frame_coords.values())
        stacked = np.vstack(matrices)
        transposed = stacked.T
        ones_row = np.ones((1, transposed.shape[1]))
        matrix_4xn = self.transform @ np.vstack([transposed, ones_row])
        new_matrix_3xn = matrix_4xn[:3, :]
        return {
            key: new_matrix_3xn[:, i : i + 1]
            for i, key in enumerate(camera_frame_coords.keys())
        }

    async def transition(self) -> None:
        """Transition to a new state and execute its associated method."""
        if self.state not in self.states:
            raise ValueError(f"Invalid state: {self.state}")
        await self.states[self.state]()

    async def calibrating(self) -> None:
        logger.info("Calibrating robot...")
        async with self.lock:
            for key, value in self.object_coord_camera_frame.items():
                if key in keurig_labels and key not in self.cache_object_coord_data:
                    self.cache_object_coord_data[key] = value

            if len(self.cache_object_coord_data) == len(keurig_labels):
                self.state = State.CALIBRATED

    async def calibrated(self) -> None:
        async with self.lock:
            self.object_coord_robot_frame = self.__transform_coordinates(
                self.object_coord_camera_frame
            )
            self.state = State.PRESSING_POWER_BUTTON

    async def pressing_power_button(self) -> None:
        logger.info("Pressin power button...")
        async with self.lock:
            power_on_target_position = self.object_coord_robot_frame["12oz"]
            ik_results = self.robot_chain.inverse_kinematics(
                power_on_target_position.reshape(1, 3)
            )
            ik_results_list = (
                ik_results.tolist()
                if isinstance(ik_results, np.ndarray)
                else ik_results
            )
            # Encode ik data
            encoded_data = json.dumps(ik_results_list).encode()
            logger.info("Publishing data to NATS: ", ik_results_list)
            await self.js.publish("robot.ik", encoded_data)
            self.state = State.PRESSED_POWER_BUTTON

    async def pressed_power_button(self) -> None:
        logger.info("Pressed power button")
        async with self.lock:
            self.state = State.GOING_HOME

    async def going_home(self) -> None:
        logger.info("Going home...")
        async with self.lock:
            return
            # match self.prev_state:
            #     case State.PRESSED_POWER_BUTTON:
            #         self.
            #     case _:
            #         return
            #
            # self.state = State.BACK_AT_HOME

    async def back_at_home(self) -> None:
        logger.info("Back at home")
        async with self.lock:
            self.state = State.STOPPED

    async def run(self) -> None:
        await self.transition()

    async def stop(self) -> None:
        return
