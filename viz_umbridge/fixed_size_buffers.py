import numpy as np
import math

__all__ = ["FixedSizeObjectBuffer", "FixedSizeFloatBuffer"]

class FixedSizeFloatBuffer:
    """
    A fixed-size buffer for storing float values. Once the buffer is full,
    new values overwrite the oldest ones in a circular manner.
    
    Attributes:
        n (int): Maximum capacity of the buffer.
        buffer (np.ndarray): Array to store the float values.
        placeholder (float): Placeholder value for empty slots.
        next_index (int): Index where the next element will be inserted.
        is_full (bool): Flag to check if the buffer has been filled at least once.
    """
    
    def __init__(self, n, placeholder=np.nan):
        self.n = n  # Maximum capacity of the buffer
        self.buffer = np.full(n, np.nan)  # Initialize with NaNs
        self.placeholder = placeholder  # Placeholder value for empty slots
        self.next_index = 0  # Index where the next element will be inserted
        self.is_full = False  # Flag to check if the buffer has been filled at least once

    def add(self, value):
        """
        Add a new float to the buffer.
        
        Args:
            value (float): The float value to be added to the buffer.
        """
        self.buffer[self.next_index] = value
        self.next_index = (self.next_index + 1) % self.n
        if self.next_index == 0:
            self.is_full = True  # Buffer has wrapped around at least once

    def get_values(self):
        """
        Retrieve buffer contents in the correct order (oldest to newest).
        
        Returns:
            np.ndarray: Array of float values in the buffer.
        """
        if not self.is_full:
            # Buffer hasn't wrapped around yet; return up to the current index
            return self.buffer[:self.next_index]
        else:
            # Buffer is full; return elements starting from next_index to the end, then from start to next_index
            return np.concatenate((self.buffer[self.next_index:], self.buffer[:self.next_index]))
        
    def get_index(self):
        return self.next_index        

    def __repr__(self):
        """
        Return a string representation of the buffer contents.
        
        Returns:
            str: String representation of the buffer.
        """
        return f"{self.get_values()}"
    

class FixedSizeObjectBuffer:
    def __init__(self, n, placeholder=None):
        self.n = n  # Maximum capacity of the buffer
        self.buffer = [placeholder] * n  # Initialize with placeholders
        self.placeholder = placeholder
        self.next_index = 0  # Index where the next element will be inserted
        self.is_full = False  # Flag to check if the buffer has been filled at least once

    def add(self, obj):
        """Add a new object to the buffer."""
        self.buffer[self.next_index] = obj
        self.next_index = (self.next_index + 1) % self.n
        if self.next_index == 0:
            self.is_full = True  # Buffer has wrapped around at least once

    def get_values(self):
        """Retrieve buffer contents in the correct order (oldest to newest)."""
        if not self.is_full:
            # Buffer hasn't wrapped around yet; return up to the current index
            return self.buffer[:self.next_index]
        else:
            # Buffer is full; return elements starting from next_index to the end, then from start to next_index
            return self.buffer[self.next_index:] + self.buffer[:self.next_index]
        
    def get_index(self):
        return self.next_index

    def __repr__(self):
        return f"{self.get_values()}"