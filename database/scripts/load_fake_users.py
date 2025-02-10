import json


if __name__ == "__main__":
    users = []
    for i in range(1, 2001):
        users.append({'id': i, 'username': f'user{i}', 'email': f"user{i}@fake.com"})

    with open('./database/students.json', 'w') as f:
        json.dump(users, f, indent=4)
