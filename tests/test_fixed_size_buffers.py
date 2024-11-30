
import unittest
import numpy as np
from viz_umbridge.fixed_size_buffers import FixedSizeFloatBuffer
from viz_umbridge.fixed_size_buffers import FixedSizeObjectBuffer

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

class TestFixedSizeObjectBuffer(unittest.TestCase):

    def test_initialization(self):
        buffer = FixedSizeObjectBuffer(5, placeholder=None)
        self.assertEqual(buffer.n, 5)
        self.assertEqual(buffer.buffer, [None] * 5)
        self.assertEqual(buffer.next_index, 0)
        self.assertFalse(buffer.is_full)

    def test_add_and_get_values(self):
        buffer = FixedSizeObjectBuffer(3, placeholder=None)
        buffer.add('a')
        buffer.add(2)
        self.assertEqual(buffer.get_values(), ['a', 2])
        
        buffer.add(3)
        self.assertEqual(buffer.get_values(), ['a', 2, 3])
        
        buffer.add(4)
        self.assertEqual(buffer.get_values(), [2, 3, 4])
        
        buffer.add(5)
        self.assertEqual(buffer.get_values(), [3, 4, 5])

    def test_add_with_placeholder(self):
        buffer = FixedSizeObjectBuffer(3, placeholder=0)
        buffer.add(1)
        buffer.add(2)
        self.assertEqual(buffer.get_values(), [1, 2])
        
        buffer.add(3)
        self.assertEqual(buffer.get_values(), [1, 2, 3])
        
        buffer.add(4)
        self.assertEqual(buffer.get_values(), [2, 3, 4])
        
        buffer.add(5)
        self.assertEqual(buffer.get_values(), [3, 4, 5])

    def test_repr(self):
        buffer = FixedSizeObjectBuffer(3, placeholder=None)
        buffer.add('a')
        buffer.add(2)
        self.assertEqual(repr(buffer), "['a', 2]")
        
        buffer.add(3)
        self.assertEqual(repr(buffer), "['a', 2, 3]")
        
        buffer.add(4)
        self.assertEqual(repr(buffer), "[2, 3, 4]")

if __name__ == '__main__':
    unittest.main()