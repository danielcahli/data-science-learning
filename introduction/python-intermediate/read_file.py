'''Students Points Aggregator
Parses a plain‑text file of student scores, validates each line, and prints total points per student.
Each line must contain exactly three whitespace‑separated fields:'''


class StudentsDataException(Exception):
    """Base custom exception for student-data processing."""
    pass


class BadLine(StudentsDataException):
    """Raised when a line in the input file is malformed."""

    def __init__(self, line_number, line_string):
        super().__init__()
        self.line_number = line_number   # 1-based line number for readability
        self.line_string = line_string   # raw offending line

    def __str__(self):
        # Friendly message when printed/logged
        return f"Bad line #{self.line_number} in source file:\n{self.line_string}"


class FileEmpty(StudentsDataException):
    """Raised when the input file is empty."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Source file empty."


from os import strerror

data = {}  # Aggregated points per student: { "First Last": total_points }

file_name = input("Enter student's data filename: ")

try:
    # Open the file for reading text
    with open(file_name, "rt") as f:
        lines = f.readlines()

    # Validate: empty file is an error
    if len(lines) == 0:
        raise FileEmpty()

    # Process each line
    for i, line in enumerate(lines):
        line = line.strip()        # remove leading/trailing whitespace & newline
        columns = line.split()     # split on whitespace

        # Expect exactly: FirstName LastName Points
        if len(columns) != 3:
            raise BadLine(i + 1, line)

        # Build the student key
        student = columns[0] + ' ' + columns[1]

        # Parse numeric points
        try:
            points = float(columns[2])
        except ValueError:
            # Re-raise as domain error with context
            raise BadLine(i + 1, line)

        # Accumulate totals
        data[student] = data.get(student, 0.0) + points

    # Output, sorted by student name
    for student in sorted(data.keys()):
        print(student, '\t', data[student])

# OS / filesystem issues (e.g., file not found, permissions)
except IOError as e:
    print("I/O error occurred:", strerror(e.errno))

# Domain validation issues (empty file, malformed line, bad points)
except StudentsDataException as e:
    print(e)
