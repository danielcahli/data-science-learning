# Example of Object-Oriented Programming (OOP) with custom exception handling.
# This program defines a custom weekday handler that can add or subtract days.
# It demonstrates encapsulation, error handling with try/except, and modular methods.

# Define a custom exception for invalid weekday input
class WeekDayError(Exception):
    pass


class Weeker:
    # Private class attribute containing the valid weekday abbreviations
    __names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(self, day):
        """
        Constructor for Weeker class.
        Initializes the object with the given weekday string (e.g., 'Mon').
        Raises WeekDayError if the string is not a valid weekday abbreviation.
        """
        try:
            # Find the index of the given day in the __names list
            self.__current = Weeker.__names.index(day)
        except ValueError:
            # If the input is invalid (not in list), raise custom exception
            raise WeekDayError

    def __str__(self):
        """
        String representation of the current weekday.
        Allows printing the object directly, returning the weekday abbreviation.
        """
        return Weeker.__names[self.__current]

    def add_days(self, n):
        """
        Advances the current weekday by 'n' days.
        Uses modulo 7 to wrap around the week if needed.
        Example: Monday + 15 days → Tuesday
        """
        self.__current = (self.__current + n) % 7

    def subtract_days(self, n):
        """
        Moves the current weekday backwards by 'n' days.
        Uses modulo 7 to wrap around correctly.
        Example: Monday - 23 days → Tuesday
        """
        self.__current = (self.__current - n) % 7


# Demonstration of class usage with error handling
try:
    weekday = Weeker('Mon')        # Initialize with Monday
    print(weekday)                 # Output: Mon

    weekday.add_days(15)           # Advance by 15 days
    print(weekday)                 # Output: Tue

    weekday.subtract_days(23)      # Move back by 23 days
    print(weekday)                 # Output: Tue

    weekday = Weeker('Monday')     # Invalid input (full name, not abbreviation)
except WeekDayError:
    # Handle invalid input gracefully
    print("Sorry, I can't serve your request.")
