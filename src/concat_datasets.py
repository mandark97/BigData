import pandas as pd

MATH_CSV = "student-alcohol-consumption/student-mat.csv"
POR_CSV = "student-alcohol-consumption/student-mat.csv"

math = pd.read_csv(MATH_CSV)
por = pd.read_csv(POR_CSV)
math['course'] = 'math'
por['course'] = 'por'

result = pd.concat([math, por])
result.to_csv("student-alcohol-consumption/students.csv")
