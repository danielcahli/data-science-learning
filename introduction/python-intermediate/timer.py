# Timer class implementation
# This program simulates a digital clock that can move forward or backward

def two_digits(val):
    s = str(val)
    if len(s) == 1:
        s = '0' + s
    return s
    """Return an integer formatted as two digits (e.g., 5 -> '05')."""


class Timer:
    def __init__(self, hours=0, minutes=0, seconds=0):
        self.__hours = hours
        self.__minutes = minutes
        self.__seconds = seconds
        
    """Return the current time as a string in HH:MM:SS format."""
    def __str__(self):
        return two_digits(self.__hours) + ":" + \
               two_digits(self.__minutes) + ":" + \
               two_digits(self.__seconds)
    
    """Advance the timer by one second, with wrap-around at 60s, 60m, 24h."""
    def next_second(self):
        self.__seconds += 1
        if self.__seconds > 59:
            self.__seconds = 0
            self.__minutes += 1
            if self.__minutes > 59:
                self.__minutes = 0
                self.__hours += 1
                if self.__hours > 23:
                    self.__hours = 0
    """Move the timer back by one second, handling wrap-around properly."""
    def prev_second(self):
        self.__seconds -= 1
        if self.__seconds < 0:
            self.__seconds = 59
            self.__minutes -= 1
            if self.__minutes < 0:
                self.__minutes = 59
                self.__hours -= 1
                if self.__hours < 0:
                    self.__hours = 23

# Example usage
timer = Timer(23, 59, 59)
print(timer)
timer.next_second()
print(timer)
timer.prev_second()
print(timer)
