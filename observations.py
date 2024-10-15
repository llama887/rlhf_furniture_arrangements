import numpy as np


class RoomObservations:
    def __init__(self, np_array, max_room_width, max_room_length, max_room_height):
        self.width = np_array[0]
        self.length = np_array[1]
        self.height = np_array[2]
        self.style = np_array[3]
        self.type = np_array[4]
        self.max_room_width = max_room_width
        self.max_room_length = max_room_length
        self.max_room_height = max_room_height

    def get_unnormalized_array(self):
        return np.array([self.width, self.length, self.height, self.style, self.type])

    def get_normalized_array(self):
        def normalize(value, max):
            normalized = 2 * value / max - 1
            assert (
                -1 <= normalized and 1 >= normalized
            ), f"The normalized value of {value} given a max of {max} is {normalized} and is not in the range -1 to 1"
            return normalized

        return np.array(
            [
                normalize(self.width, self.max_room_width),
                normalize(self.length, self.max_room_width),
                normalize(self.height, self.max_room_width),
                self.style,  # dont normalize categorical values
                self.type,
            ],
            dtype=np.float32,
        )


class FurnitureObservations:
    class Furniture:
        def __init__(self, id, style, width, height, depth, x, y, theta):
            self.id = id
            self.style = style
            self.width = width
            self.height = height
            self.depth = depth
            self.x = x
            self.y = y
            self.theta = theta

        def get_unnormalized_array(self):
            return np.array(
                [
                    self.id,
                    self.style,
                    self.width,
                    self.height,
                    self.depth,
                    self.x,
                    self.y,
                    self.theta,
                ],
                dtype=np.float32,
            )

        def get_normalized_array(
            self,
            max_furniture_width,
            max_furniture_height,
            max_furniture_depth,
            max_furniture_x,
            max_furniture_y,
        ):
            def normalize(value, max_value):
                normalized = 2 * value / max_value - 1
                assert (
                    -1 <= normalized <= 1
                ), f"The normalized value of {value} given a max of {max_value} is {normalized} and is not in the range -1 to 1"
                return normalized

            return np.array(
                [
                    self.id,  # Don't normalize id (categorical)
                    self.style,  # Don't normalize style (categorical)
                    normalize(self.width, max_furniture_width),
                    normalize(self.height, max_furniture_height),
                    normalize(self.depth, max_furniture_depth),
                    normalize(self.x, max_furniture_x),
                    normalize(self.y, max_furniture_y),
                    normalize(self.theta, 360),
                ]
            )

    def __init__(
        self,
        np_array,
        max_furniture_id,
        max_furniture_style,
        max_furniture_width,
        max_furniture_depth,
        max_furniture_height,
        max_furniture_x,
        max_furniture_y,
    ):
        self.furnitures = [
            self.Furniture(
                np_array[i],
                np_array[i + 1],
                np_array[i + 2],
                np_array[i + 3],
                np_array[i + 4],
                np_array[i + 5],
                np_array[i + 6],
                np_array[i + 7],
            )
            for i in range(0, len(np_array), 8)
        ]
        self.max_furniture_id = max_furniture_id
        self.max_furniture_style = max_furniture_style
        self.max_furniture_width = max_furniture_width
        self.max_furniture_depth = max_furniture_depth
        self.max_furniture_height = max_furniture_height
        self.max_furniture_x = max_furniture_x
        self.max_furniture_y = max_furniture_y

    def get_unnormalized_array(self):
        return np.array(
            [furniture.get_unnormalized_array() for furniture in self.furnitures]
        )

    def get_normalized_array(self):
        return np.array(
            [
                furniture.get_normalized_array(
                    self.max_furniture_width,
                    self.max_furniture_height,
                    self.max_furniture_depth,
                    self.max_furniture_x,
                    self.max_furniture_y,
                )
                for furniture in self.furnitures
            ]
        )
