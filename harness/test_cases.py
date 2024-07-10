TESTS_LITE = {
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

TESTS = {
    ## Easy Tests
    # Simple orders
    '1': lambda x, y, z: x > y > z,
    '2': lambda x, y, z: x < y < z,
    '3': lambda x, y, z: x >= y >= z,
    '4': lambda x, y, z: x <= y <= z,
    '5': lambda x, y, z: x < z < y,
    '6': lambda x, y, z: x <= z <= y,
    '7': lambda x, y, z: z < x < y,
    '8': lambda x, y, z: x <= x <= y,
    # Equality and Inequality
    '9': lambda x, y, z: x == y and y == z,
    '10': lambda x, y, z: x != y and y != z and x != z,
    # Signs 
    '11': lambda x, y, z: x < 0 and y < 0 and z < 0,
    '12': lambda x, y, z: x > 0 and y > 0 and z > 0,
    # Even vs Odd
    '13': lambda x, y, z: (x % 2 == 0) and (y % 2 == 0) and (z % 2 == 0),
    '14': lambda x, y, z: (x % 2 == 1) and (y % 2 == 1) and (z % 2 == 1),
    # Arithmetic (no division)
    '15': lambda x, y, z: x + y == z,
    '16': lambda x, y, z: x * y == z,
    '17': lambda x, y, z: x - y == z,
    '18': lambda x, y, z: x + z == y,
    '19': lambda x, y, z: x * z == y,
    '20': lambda x, y, z: x - z == y,
    '21': lambda x, y, z: y + z == x,
    '22': lambda x, y, z: y * z == x,
    '23': lambda x, y, z: y - z == x,
    # Reversed order for subtraction
    '24': lambda x, y, z: y - x == z,
    '25': lambda x, y, z: z - x == y,
    '26': lambda x, y, z: z - y == x,
    ## Harder Tests
    # Pythagorean Triple
    '27': lambda x, y, z: x**2 + y**2 == z**2,
    '28': lambda x, y, z: x**2 + z**2 == y**2,
    '29': lambda x, y, z: y**2 + z**2 == x**2,
    # Bitwise Operations
    '30': lambda x, y, z: (x & y) == z,
    '31': lambda x, y, z: (x | y) == z,
    '32': lambda x, y, z: (x ^ y) == z,
    # Coprimality
    '33': lambda x, y, z: all(__gcd(a, b) == 1 for a, b in [(x, y), (y, z), (z, x)]),
    # Perfect Squares
    '34': lambda x, y, z: all(int(n**0.5)**2 == n for n in (x, y, z)),
    # Sums / Products
    '35': lambda x, y, z: (x + y + z) == 0,
    '36': lambda x, y, z: (x * y * z) == 0,
    '37': lambda x, y, z: (x + y + z) % 2 == 0,
    '38': lambda x, y, z: (x + y + z) % 2 == 1,
    '39': lambda x, y, z: (x * y * z) % 2 == 0,
    '40': lambda x, y, z: (x * y * z) % 2 == 1,
    # Max / Min
    '41': lambda x, y, z: max(x, y, z) == x,
    '42': lambda x, y, z: max(x, y, z) == y,
    '43': lambda x, y, z: max(x, y, z) == z,
    '44': lambda x, y, z: min(x, y, z) == x,
    '45': lambda x, y, z: min(x, y, z) == y,
    '46': lambda x, y, z: min(x, y, z) == z,
    # Evil Float Cases
    '47': lambda x, y, z: (0 < x % 1) and (0 < y % 1) and (0 < z % 1),
    '48': lambda x, y, z: 0 < x % 1 < y % 1 < z % 1 < 1,
    '49': lambda x, y, z: (x < y < z) and (0 < z - x <= 1),
    # Arithmetic Mean
    '50': lambda x, y, z: (x + y) / 2 == z,
}
