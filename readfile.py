class StudentsDataException(Exception):
    pass


class BadLine(StudentsDataException):
    def __init__(self, line_number, line_string):
        super().__init__()
        self.line_number = line_number
        self.line_string = line_string

    def __str__(self):
        return f"Bad line #{self.line_number} in source file:\n{self.line_string}"


class FileEmpty(StudentsDataException):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Source file empty."


from os import strerror

data = {}

file_name = input("Enter student's data filename: ")

try:
    with open(file_name, "rt") as f:
        lines = f.readlines()

    if len(lines) == 0:
        raise FileEmpty()

    for i, line in enumerate(lines):
        line = line.strip()
        columns = line.split()

        if len(columns) != 3:
            raise BadLine(i + 1, line)

        student = columns[0] + ' ' + columns[1]

        try:
            points = float(columns[2])
        except ValueError:
            raise BadLine(i + 1, line)

        data[student] = data.get(student, 0) + points

    for student in sorted(data.keys()):
        print(student, '\t', data[student])

except IOError as e:
    print("I/O error occurred:", strerror(e.errno))
except StudentsDataException as e:
    print(e)
