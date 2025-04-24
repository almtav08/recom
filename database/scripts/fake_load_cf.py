import json
import random
import time
import sys

sys.path.append(".")
from database.orm.query_generator import QueryGenerator
from database.models.db import Interaction

if __name__ == "__main__":
    client = QueryGenerator()
    client.connect()
    # resources = client.list_resources()
    users = client.list_users()

    path_a = [2, 5, 7, 8, 11, 15, 17, 18, 21, 24, 29] # Ve solo lo obligatorio # Aprueba
    path_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29] # Ve todo # Aprueba
    path_c = [28, 2, 9, 20, 5, 13, 16, 22, 27, 26, 1, 11] # Ve un numero aleatorio de items y los items son aleatorios (generados con random) # Suspende
    path_d = [2, 5, 8, 7, 11, 5, 7, 15, 17, 18, 21, 24, 15, 21, 17, 29] # Ve solo lo obligatorio pero repite items # Aprueba
    path_e = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 7, 6, 5, 2, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 15, 17, 20, 23, 22, 25, 26, 27, 28, 29] # Ve todo pero repite items # Aprueba
    path_f = [2, 3, 5, 7, 8, 12, 15, 17, 20, 22, 26, 27] # Ve un poco de poco # Suspende
    path_g = [9, 16, 23, 17, 22, 5, 8, 10, 27, 15, 24, 21, 28, 11, 25, 14, 26, 4, 7] # Ve un numero aleatorio de items y los items son aleatorios (generados con random) pero más que el random anterior # Suspende
    options = ['path_a', 'path_b', 'path_c', 'path_d', 'path_e', 'path_f', 'path_g']

    user_grades = {}
    for user in users:
        path = random.choice(options)
        user_grades[user.id] = "Pass" if path in ['path_a', 'path_b', 'path_d', 'path_e'] else "Fail"
        path = eval(path)
        user_path = path.copy()
        # Add the chance to change to permute two items in the path
        rand_idx = -1
        if random.random() < 0.6:
            index = random.randint(0, len(user_path) - 2)
            if index + 1 < len(user_path):
                rand_idx = index
                user_path[index], user_path[index + 1] = user_path[index + 1], user_path[index]
        random_change = False
        for item in user_path:
            # Add the chance to change to a random item
            if item != 29 and random.random() < 0.5 and not random_change and rand_idx != item:
                item = random.randint(1, 28)
                random_change = True
            client.insert_interaction(Interaction(timestamp=int(time.time()), user_id=user.id, resource_id=item))

    with open(f"./database/data/user_grades.json", 'w') as f:
        json.dump(user_grades, f)

    client.disconnect()
