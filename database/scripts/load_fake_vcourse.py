import numpy as np
import pandas as pd
import json
import time
import sys
import matplotlib.pyplot as plt
import random
from collections import defaultdict, Counter, deque
from tqdm import tqdm

sys.path.append(".")
from database.orm.query_generator import QueryGenerator
from database.models.db import Interaction
from generators.markov_generator import MarkovPathGenerator


if __name__ == "__main__":

    v_logs = pd.read_csv("log_data/vcourse.csv", date_format="%Y-%m-%d %H:%M:%S")
    with open("database/data/vcourse/resources.json", "r") as f:
        resources = json.load(f)
    resources = {res["name"]: res["recid"] for res in resources}

    student_results = {
        "180599": "Fail",
        "163849": "Fail",
        "180545": "Fail",
        "159733": "Fail",
        "186155": "Fail",
        "185006": "Pass",
        "186158": "Fail",
        "179682": "Fail",
        "179356": "Fail",
        "140032": "Fail",
        "185004": "Pass",
        "186479": "Fail",
        "194146": "Fail",
        "184985": "Fail",
        "186153": "Fail",
        "184996": "Pass",
        "178614": "Pass",
        "185003": "Pass",
        "185000": "Fail",
        "185673": "Pass",
        "186480": "Fail",
        "184986": "Pass",
        "181091": "Fail",
        "186161": "Fail",
        "141174": "Pass",
        "184980": "Pass",
        "171442": "Fail",
        "140230": "Fail",
        "186162": "Fail",
        "186157": "Fail",
        "186152": "Fail",
        "186478": "Pass",
        "186164": "Pass",
        "185011": "Fail",
        "185009": "Pass",
        "184999": "Fail",
        "184992": "Fail",
        "184988": "Pass",
        "184990": "Fail",
        "184885": "Pass",
        "184983": "Fail",
        "184994": "Fail",
        "170619": "Fail",
        "184998": "Fail",
        "193349": "Fail",
        "185674": "Fail",
        "159551": "Fail",
        "141444": "Fail",
        "174057": "Fail",
    }

    v_logs["item"] = v_logs["item"].apply(
        lambda x: x.replace("URL: ", "")
        .replace("Archivo: ", "")
        .replace("Tarea: ", "")
        .replace("Foro: ", "")
        .replace("Carpeta: ", "")
    )

    not_in = []
    for item in v_logs["item"].unique():
        if item not in resources.keys():
            not_in.append(item)

    v_logs = v_logs[~v_logs["item"].isin(not_in)]
    v_logs["item"] = v_logs["item"].apply(
        lambda x: resources[x] if x in resources else x
    )
    v_logs_red = v_logs[["user", "item", "time"]].copy()

    user_lists = {}
    for user in v_logs_red["user"].unique():
        user_logs = v_logs_red[v_logs_red["user"] == user]
        user_logs = user_logs.sort_values(by="time")
        user_lists[str(user)] = user_logs["item"].tolist()

    for user in user_lists:
        items = user_lists[user]
        filtered_items = []
        last_item = None
        for item in items:
            if item != last_item:
                filtered_items.append(item)
                last_item = item
        user_lists[user] = filtered_items

    pass_paths = []
    pass_length = []
    fail_paths = []
    fail_length = []

    for user, result in student_results.items():
        if result == "Pass":
            pass_paths.append(user_lists[user])
            pass_length.append(len(user_lists[user]))
        else:
            fail_paths.append(user_lists[user])
            fail_length.append(len(user_lists[user]))

    avg_pass_length = np.mean(pass_length) if pass_paths else 1
    std_pass_length = np.std(pass_length) if pass_paths else 0

    avg_fail_length = np.mean(fail_length) if fail_paths else 1
    std_fail_length = np.std(fail_length) if fail_paths else 0

    pass_generator = MarkovPathGenerator(final_exam_id=120, order=2)
    fail_generator = MarkovPathGenerator(final_exam_id=120, order=2)

    pass_generator.train(pass_paths)
    fail_generator.train(fail_paths)

    client = QueryGenerator()
    client.connect()
    users = client.list_users()

    real_pass_users = [
        student for student, result in student_results.items() if result == "Pass"
    ]
    real_fail_users = [
        student for student, result in student_results.items() if result == "Fail"
    ]

    n_pass_users = int(len(real_pass_users) +21)
    n_fail_users = len(users) - n_pass_users

    print(
        f"Generando {n_pass_users - len(real_pass_users)} usuarios aprobados y {n_fail_users - len(real_fail_users)} usuarios suspendidos..."
    )

    user_grades = {}
    for i in tqdm(range(len(users))):
        interactions = []
        if i < n_pass_users:
            if len(real_pass_users) > 0:
                user = real_pass_users.pop(0)
                for item in user_lists[user]:
                    interactions.append(
                        Interaction(
                            timestamp=int(time.time()),
                            user_id=users[i].id,
                            resource_id=item,
                        )
                    )
            else:
                path = pass_generator.generate_path(
                    random.randint(
                        int(avg_pass_length - std_pass_length),
                        int(avg_pass_length + std_pass_length),
                    )
                )
                for item in path:
                    interactions.append(
                        Interaction(
                            timestamp=int(time.time()),
                            user_id=users[i].id,
                            resource_id=item,
                        )
                    )
            user_grades[users[i].id] = "Pass"
        else:
            if len(real_fail_users) > 0:
                user = real_fail_users.pop(0)
                for item in user_lists[user]:
                    interactions.append(
                        Interaction(
                            timestamp=int(time.time()),
                            user_id=users[i].id,
                            resource_id=item,
                        )
                    )
            else:
                path = fail_generator.generate_path(
                    random.randint(
                        int(avg_fail_length - std_fail_length),
                        int(avg_fail_length + std_fail_length),
                    )
                )
                for item in path:
                    interactions.append(
                        Interaction(
                            timestamp=int(time.time()),
                            user_id=users[i].id,
                            resource_id=item,
                        )
                    )
            user_grades[users[i].id] = "Fail"
        client.insert_interactions(interactions)
    client.disconnect()

    with open("database/data/user_grades.json", "w") as f:
        json.dump(user_grades, f)
