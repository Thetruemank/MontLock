"""
MontLock Cursor Locker Module

This module implements the cursor locking mechanism for Windows.
When unauthorized use is detected, it freezes the cursor in place
for a specified duration.
"""

import time
import threading
import ctypes
from ctypes import wintypes
import sys

# Windows API constants and functions
user32 = ctypes.WinDLL('user32', use_last_error=True)

# Define required Windows API functions
user32.GetCursorPos = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    ctypes.POINTER(wintypes.POINT)
)(("GetCursorPos", user32))

user32.SetCursorPos = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.INT,
    wintypes.INT
)(("SetCursorPos", user32))

user32.ClipCursor = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    ctypes.POINTER(wintypes.RECT)
)(("ClipCursor", user32))


class CursorLocker:
    def __init__(self):
        """Initialize the cursor locker."""
        self.lock_thread = None
        self.is_locked = False
        self.lock_event = threading.Event()
    
    def _get_cursor_pos(self):
        """
        Get the current cursor position.
        
        Returns:
            tuple: (x, y) coordinates of the cursor
        """
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        return (point.x, point.y)
    
    def _set_cursor_pos(self, x, y):
        """
        Set the cursor position.
        
        Args:
            x (int): X-coordinate
            y (int): Y-coordinate
            
        Returns:
            bool: True if successful, False otherwise
        """
        return user32.SetCursorPos(x, y)
    
    def _clip_cursor(self, rect=None):
        """
        Confine the cursor to a rectangular area of the screen.
        
        Args:
            rect (tuple, optional): (left, top, right, bottom) or None to remove clipping
            
        Returns:
            bool: True if successful, False otherwise
        """
        if rect is None:
            return user32.ClipCursor(None)
        
        r = wintypes.RECT()
        r.left, r.top, r.right, r.bottom = rect
        return user32.ClipCursor(ctypes.byref(r))
    
    def _lock_cursor_thread(self, x, y, duration):
        """
        Thread function that keeps the cursor locked at a specific position.
        
        Args:
            x (int): X-coordinate to lock the cursor at
            y (int): Y-coordinate to lock the cursor at
            duration (float): Duration in seconds to keep the cursor locked
        """
        print(f"Cursor locked at position ({x}, {y}) for {duration} seconds.")
        
        # Create a small rectangle around the cursor position
        rect = (x-2, y-2, x+2, y+2)
        
        # Set the flag to indicate that the cursor is locked
        self.is_locked = True
        
        # Record the start time
        start_time = time.time()
        
        try:
            # Clip the cursor to a small area
            self._clip_cursor(rect)
            
            # Keep setting the cursor position until the duration expires or lock is released
            while time.time() - start_time < duration and not self.lock_event.is_set():
                self._set_cursor_pos(x, y)
                time.sleep(0.01)  # Small sleep to reduce CPU usage
        
        finally:
            # Release the cursor clipping
            self._clip_cursor(None)
            
            # Reset the flag
            self.is_locked = False
            self.lock_event.clear()
            
            print("Cursor unlocked.")
    
    def lock_cursor(self, duration=10, position=None):
        """
        Lock the cursor at the current position or a specified position.
        
        Args:
            duration (float): Duration in seconds to keep the cursor locked
            position (tuple, optional): (x, y) position to lock the cursor at
        """
        # If already locked, do nothing
        if self.is_locked:
            print("Cursor is already locked.")
            return
        
        # Get current position if not specified
        if position is None:
            position = self._get_cursor_pos()
        
        x, y = position
        
        # Stop any existing lock thread
        if self.lock_thread and self.lock_thread.is_alive():
            self.lock_event.set()
            self.lock_thread.join(timeout=1.0)
        
        # Start a new lock thread
        self.lock_event.clear()
        self.lock_thread = threading.Thread(
            target=self._lock_cursor_thread,
            args=(x, y, duration)
        )
        self.lock_thread.daemon = True
        self.lock_thread.start()
    
    def unlock_cursor(self):
        """
        Unlock the cursor if it's currently locked.
        """
        if not self.is_locked:
            print("Cursor is not locked.")
            return
        
        # Signal the lock thread to stop
        self.lock_event.set()
        
        # Wait for the thread to finish
        if self.lock_thread and self.lock_thread.is_alive():
            self.lock_thread.join(timeout=1.0)
            
        print("Cursor unlocked.")


if __name__ == "__main__":
    # Example usage
    locker = CursorLocker()
    
    print("Locking cursor for 5 seconds...")
    locker.lock_cursor(duration=5)
    
    # Wait for the lock to be released
    time.sleep(6)
    
    print("Test complete.") 