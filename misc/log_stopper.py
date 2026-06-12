import os
import sys

class LogStopper:
    """Forcefully suppresses OS-level stdout and stderr file descriptors."""
    def __enter__(self):
        # Flush Python streams to avoid cutting off pending prints
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Open the null device
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        
        # Duplicate the original file descriptors to save them
        self.orig_stdout_fd = os.dup(1)
        self.orig_stderr_fd = os.dup(2)
        
        # Overwrite file descriptors 1 and 2 with the null device
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original file descriptors
        os.dup2(self.orig_stdout_fd, 1)
        os.dup2(self.orig_stderr_fd, 2)
        
        # Clean up by closing the duplicates
        os.close(self.null_fd)
        os.close(self.orig_stdout_fd)
        os.close(self.orig_stderr_fd)