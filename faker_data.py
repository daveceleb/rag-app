from faker import Faker
import pandas as pd
import random
from datetime import datetime, timedelta

fake = Faker()

students_per_year = 200
years = range(1990, 2026)

data = []

student_counter = 10000  # starting ID

for year in years:
    for _ in range(students_per_year):
        admission_year = year
        degree_level = random.choice(["BSc", "BA", "MSc", "MA", "PhD"])
        duration = {"BSc": 4, "BA": 4, "MSc": 2, "MA": 2, "PhD": 5}[degree_level]
        graduation_year = admission_year + duration

        cgpa = round(random.uniform(2.0, 4.0), 2)
        status = random.choice(["Graduated", "Ongoing", "Dropped"])
        degree_class = random.choice(["First Class", "Second Class Upper", "Second Class Lower", "Pass"])
        credits_completed = random.randint(0, 180)
        scholarship = random.choice(["Excellence Scholarship", "Merit Scholarship", "None"])
        residential_status = random.choice(["On-Campus", "Off-Campus"])
        funding_type = random.choice(["Self-funded", "Government-funded", "Private-sponsored"])
        
        data.append({
            "student_id": student_counter,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "gender": random.choice(["Male", "Female"]),
            "birth_date": fake.date_of_birth(minimum_age=17, maximum_age=35),
            "region_of_origin": fake.state(),
            "campus": random.choice(["Main Campus", "City Campus", "North Campus"]),
            "degree_level": degree_level,
            "programme": fake.job(),
            "department": random.choice(["Computer Science", "Business", "Engineering", "Law", "Arts"]),
            "faculty": random.choice(["Science", "Business", "Engineering", "Law", "Arts"]),
            "admission_year": admission_year,
            "graduation_year": graduation_year,
            "status": status,
            "cgpa": cgpa,
            "degree_class": degree_class,
            "credits_completed": credits_completed,
            "residential_status": residential_status,
            "funding_type": funding_type,
            "scholarship_name": scholarship if scholarship != "None" else "",
            "last_update": datetime.now()
        })
        student_counter += 1

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv("academic_dataset_faker.csv", index=False)
print("Fake academic dataset generated successfully!")
