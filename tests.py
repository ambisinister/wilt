TESTS = {
    '1': lambda x, y, z: x > y > z,
    '2': lambda x, y, z: x < y < z,
    '3': lambda x, y, z: x >= y >= z,
    '4': lambda x, y, z: x <= y <= z,
    '5': lambda x, y, z: x == y and y == z,
    '6': lambda x, y, z: x != y and y != z and x != z,
    '7': lambda x, y, z: x < 0 and y < 0 and z < 0,
    '8': lambda x, y, z: x + y == z,
    '9': lambda x, y, z: x * y == z,
    '10': lambda x, y, z: x < y and y > z
}
