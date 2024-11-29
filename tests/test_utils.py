
import unittest
import numpy as np
from viz_umbridge.utils import FixedSizeFloatBuffer

class TestFixedSizeFloatBuffer(unittest.TestCase):

    def test_initialization(self):
        buffer = FixedSizeFloatBuffer(5)
        self.assertEqual(buffer.n, 5)
        self.assertTrue(np.all(np.isnan(buffer.buffer)))
        self.assertEqual(buffer.next_index, 0)
        self.assertFalse(buffer.is_full)

    def test_add_and_get_values(self):
        buffer = FixedSizeFloatBuffer(3)
        buffer.add(1.0)
        buffer.add(2.0)
        self.assertTrue(np.array_equal(buffer.get_values(), [1.0, 2.0]))
        
        buffer.add(3.0)
        self.assertTrue(np.array_equal(buffer.get_values(), [1.0, 2.0, 3.0]))
        
        buffer.add(4.0)
        self.assertTrue(np.array_equal(buffer.get_values(), [2.0, 3.0, 4.0]))
        
        buffer.add(5.0)
        self.assertTrue(np.array_equal(buffer.get_values(), [3.0, 4.0, 5.0]))

    def test_repr(self):
        buffer = FixedSizeFloatBuffer(3)
        buffer.add(1.0)
        buffer.add(2.0)
        self.assertEqual(repr(buffer), "[1. 2.]")
        
        buffer.add(3.0)
        self.assertEqual(repr(buffer), "[1. 2. 3.]")
        
        buffer.add(4.0)
        self.assertEqual(repr(buffer), "[2. 3. 4.]")

if __name__ == '__main__':
    unittest.main()