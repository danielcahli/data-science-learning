import math


class Point:
    def __init__(self, x=0.0, y=0.0):
        # Private attributes for Cartesian coordinates
        self.__x = x
        self.__y = y

    def getx(self):
        """Return the x-coordinate."""
        return self.__x

    def gety(self):
        """Return the y-coordinate."""
        return self.__y

    def distance_from_xy(self, x, y):
        """Return distance from this point to coordinates (x, y)."""
        return math.hypot(self.__x - x, self.__y - y)

    def distance_from_point(self, point):
        """Return distance from this point to another Point object."""
        return self.distance_from_xy(point.getx(), point.gety())


class Triangle:
    def __init__(self, vertice1, vertice2, vertice3):
        # Store three vertices (Point objects) in a list
        self.__vertices = [vertice1, vertice2, vertice3]

    def perimeter(self):
        """Return the perimeter of the triangle (sum of side lengths)."""
        per = 0
        for i in range(3):
            per += self.__vertices[i].distance_from_point(self.__vertices[(i + 1) % 3])
        return per


# Example usage
triangle = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
print(triangle.perimeter())  # 3.414213562373095
