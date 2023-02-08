import time

import requests


def set_velocity(vel0, vel1):
    r = requests.get(f"http://localhost:8080/robot/set/velocity?value={vel0},{vel1}")


def main() -> None:
    try:
        kd = 30
        for vel in range(2, 5):
            print(f"going velocity: {vel*kd}")
            set_velocity(vel * kd, vel * kd)
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        set_velocity(0, 0)
        print("done")


if __name__ == "__main__":
    main()
