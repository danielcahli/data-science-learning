import calendar

class MyCalendar(calendar.Calendar):
    def count_weekday_in_year(self, year, weekday):
        count = 0
        for month in range(1, 13):
            for week in self.monthdays2calendar(year, month):
                for day, wkday in week:
                    if day != 0 and wkday == weekday:
                        count += 1
        return count

my_calendar = MyCalendar()
number_of_days = my_calendar.count_weekday_in_year(2019, calendar.MONDAY)

print(number_of_days)
