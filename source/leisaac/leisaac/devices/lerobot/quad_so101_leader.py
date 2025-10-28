from collections.abc import Callable

from .so101_leader import SO101Leader
from ..device_base import Device


class QuadSO101Leader(Device):
    def __init__(
        self,
        env,
        nord_port: str = '/dev/ttyACM0',
        ost_port: str = '/dev/ttyACM1',
        west_port: str = '/dev/ttyACM2',
        sud_port: str = '/dev/ttyACM3',
        recalibrate: bool = False
    ):
        super().__init__(env)

        # Connect to 4 SO101 leaders
        print("Connecting to nord_so101_leader...")
        self.nord_so101_leader = SO101Leader(env, nord_port, recalibrate, "nord_so101_leader.json")

        print("Connecting to ost_so101_leader...")
        self.ost_so101_leader = SO101Leader(env, ost_port, recalibrate, "ost_so101_leader.json")

        print("Connecting to west_so101_leader...")
        self.west_so101_leader = SO101Leader(env, west_port, recalibrate, "west_so101_leader.json")

        print("Connecting to sud_so101_leader...")
        self.sud_so101_leader = SO101Leader(env, sud_port, recalibrate, "sud_so101_leader.json")

        # Use nord as main device for keyboard control, stop keyboard listeners on others
        self.ost_so101_leader.stop_keyboard_listener()
        self.west_so101_leader.stop_keyboard_listener()
        self.sud_so101_leader.stop_keyboard_listener()

    def __str__(self) -> str:
        """Returns: A string containing the information of quad-so101 leader."""
        msg = "Quad-SO101-Leader device for SE(3) control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove Quad-SO101-Leader to control Quad-SO101-Follower\n"
        msg += "\tIf SO101-Follower can't synchronize with Quad-SO101-Leader, please add --recalibrate and rerun to recalibrate Quad-SO101-Leader.\n"
        return msg

    def add_callback(self, key: str, func: Callable):
        # Only add callbacks to nord (main device)
        self.nord_so101_leader.add_callback(key, func)
        # Add no-op callbacks to others
        self.ost_so101_leader.add_callback(key, lambda: None)
        self.west_so101_leader.add_callback(key, lambda: None)
        self.sud_so101_leader.add_callback(key, lambda: None)

    def reset(self):
        self.nord_so101_leader.reset()
        self.ost_so101_leader.reset()
        self.west_so101_leader.reset()
        self.sud_so101_leader.reset()

    def get_device_state(self):
        return {
            "nord_arm": self.nord_so101_leader.get_device_state(),
            "ost_arm": self.ost_so101_leader.get_device_state(),
            "west_arm": self.west_so101_leader.get_device_state(),
            "sud_arm": self.sud_so101_leader.get_device_state()
        }

    def input2action(self):
        state = {}
        reset = state["reset"] = self.nord_so101_leader.reset_state
        state['started'] = self.nord_so101_leader.started
        if reset:
            self.nord_so101_leader.reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self.nord_so101_leader.started
        ac_dict['quad_so101_leader'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        ac_dict['motor_limits'] = {
            'nord_arm': self.nord_so101_leader.motor_limits,
            'ost_arm': self.ost_so101_leader.motor_limits,
            'west_arm': self.west_so101_leader.motor_limits,
            'sud_arm': self.sud_so101_leader.motor_limits
        }
        return ac_dict
